#include "llama-prune.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>

//
// Internal state for Nash equilibrium computation
//

struct llama_nash_state {
    const llama_model             & model;
    const llama_nash_prune_params & params;

    int n_layer;
    std::vector<int> n_head;      // Original head counts per layer
    std::vector<int> n_head_kv;   // Original KV head counts per layer
    int n_embd;
    int n_embd_head_k;
    int head_dim;  // Dimension per head (same as n_embd_head_k)

    // Per-head data [layer][head]
    std::vector<std::vector<float>> importance;     // c_i: contribution scores
    std::vector<std::vector<float>> participation;  // s_i: Nash equilibrium values

    // Redundancy matrix [layer][head_i][head_j]
    std::vector<std::vector<std::vector<float>>> redundancy;

    // Constructor
    llama_nash_state(const llama_model & model_, const llama_nash_prune_params & params_)
        : model(model_)
        , params(params_)
    {
        const auto & hparams = model.hparams;

        n_layer       = (int)hparams.n_layer;
        n_embd        = (int)hparams.n_embd;
        n_embd_head_k = (int)hparams.n_embd_head_k;
        head_dim      = n_embd_head_k;  // Alias for convenience

        // Initialize per-layer head counts
        n_head.resize(n_layer);
        n_head_kv.resize(n_layer);

        for (int il = 0; il < n_layer; il++) {
            n_head[il]    = (int)hparams.n_head(il);
            n_head_kv[il] = (int)hparams.n_head_kv(il);
        }

        // Allocate importance and participation arrays
        importance.resize(n_layer);
        participation.resize(n_layer);
        redundancy.resize(n_layer);

        for (int il = 0; il < n_layer; il++) {
            int nh = n_head[il];
            importance[il].resize(nh, 1.0f);
            participation[il].resize(nh, 1.0f);

            redundancy[il].resize(nh);
            for (int h = 0; h < nh; h++) {
                redundancy[il][h].resize(nh, 0.0f);
            }
        }
    }
};

//
// Importance computation
//


// Compute head importance from imatrix data
// The imatrix stores per-row importance (one value per input dimension)
// For attention output weights (wo) with shape [head_dim * n_heads, n_embd]:
//   - imatrix has (head_dim * n_heads) values
//   - Each head corresponds to head_dim consecutive input dimensions
//   - Head importance = sum of importance for that head's dimensions
static bool compute_importance_from_imatrix(
    llama_nash_state          & state,
    const llama_imatrix_data  & imatrix
) {
    LLAMA_LOG_INFO("%s: computing head importance from imatrix data\n", __func__);

    int layers_found = 0;

    for (int il = 0; il < state.n_layer; il++) {
        int n_head = state.n_head[il];
        int head_dim = state.head_dim;

        // Look for attention output weight tensor importance for this layer
        // wo has shape [head_dim * n_heads, n_embd], so imatrix has (head_dim * n_heads) values
        // These map directly to head dimensions!
        std::vector<std::string> wo_names = {
            "blk." + std::to_string(il) + ".attn_output.weight",
            "blk." + std::to_string(il) + ".attn_o_proj.weight",
            "model.layers." + std::to_string(il) + ".self_attn.o_proj.weight"
        };

        const std::vector<float> * wo_importance = nullptr;
        std::string found_name;

        for (const auto & name : wo_names) {
            auto it = imatrix.find(name);
            if (it != imatrix.end()) {
                wo_importance = &it->second;
                found_name = name;
                break;
            }
        }

        if (!wo_importance) {
            // Try pattern matching for attention output weight
            for (const auto & [name, values] : imatrix) {
                if (name.find("attn_output") != std::string::npos &&
                    name.find(".weight") != std::string::npos) {
                    std::string layer_str = "blk." + std::to_string(il) + ".";
                    if (name.find(layer_str) != std::string::npos) {
                        wo_importance = &values;
                        found_name = name;
                        break;
                    }
                }
            }
        }

        if (!wo_importance || wo_importance->empty()) {
            LLAMA_LOG_WARN("%s: no imatrix data for layer %d attention output, using heuristic\n",
                           __func__, il);
            continue;
        }

        layers_found++;

        // wo weight shape: [head_dim * n_heads, n_embd]
        // imatrix has one importance value per input dimension = head_dim * n_heads
        size_t expected_size = (size_t)(head_dim * n_head);

        if (wo_importance->size() != expected_size) {
            LLAMA_LOG_INFO("%s: layer %d wo imatrix size %zu (expected %zu)\n",
                           __func__, il, wo_importance->size(), expected_size);
        }

        // Compute importance per head by summing importance of that head's dimensions
        // Head h owns dimensions [h*head_dim, (h+1)*head_dim)
        for (int h = 0; h < n_head; h++) {
            float head_imp = 0.0f;

            for (int d = 0; d < head_dim; d++) {
                size_t idx = (size_t)(h * head_dim + d);
                if (idx < wo_importance->size()) {
                    head_imp += (*wo_importance)[idx];
                }
            }

            state.importance[il][h] = head_imp;
        }

        // Normalize importance within layer
        float sum = 0.0f;
        float min_imp = state.importance[il][0];
        float max_imp = state.importance[il][0];
        for (int h = 0; h < n_head; h++) {
            sum += state.importance[il][h];
            min_imp = std::min(min_imp, state.importance[il][h]);
            max_imp = std::max(max_imp, state.importance[il][h]);
        }

        if (sum > 0.0f) {
            for (int h = 0; h < n_head; h++) {
                state.importance[il][h] /= sum;
                state.importance[il][h] *= n_head; // Scale to ~1.0 average
            }
        }

        if (state.params.verbose) {
            LLAMA_LOG_INFO("%s: layer %d importance from '%s': [", __func__, il, found_name.c_str());
            for (int h = 0; h < std::min(n_head, 5); h++) {
                LLAMA_LOG_INFO("%.3f%s", state.importance[il][h], h < n_head-1 ? ", " : "");
            }
            if (n_head > 5) LLAMA_LOG_INFO("...");
            LLAMA_LOG_INFO("] (range: %.3f - %.3f)\n", min_imp, max_imp);
        }
    }

    LLAMA_LOG_INFO("%s: found imatrix data for %d/%d layers\n", __func__, layers_found, state.n_layer);
    return layers_found > 0;
}

// Compute head importance using heuristics when no imatrix available
static void compute_importance_heuristic(llama_nash_state & state) {
    LLAMA_LOG_INFO("%s: using heuristic importance (no calibration data)\n", __func__);

    // U-shaped importance: early and late layers tend to be more important
    for (int il = 0; il < state.n_layer; il++) {
        float layer_factor = 1.0f;
        float mid = (float)(state.n_layer - 1) / 2.0f;
        float dist = std::abs((float)il - mid) / mid;
        layer_factor = 0.7f + 0.6f * dist;  // Range: 0.7 to 1.3

        for (int h = 0; h < state.n_head[il]; h++) {
            // Add some head-based variation
            float head_factor = 1.0f + 0.1f * std::sin((float)h * 0.5f);
            state.importance[il][h] = layer_factor * head_factor;
        }

        // Normalize within layer
        float sum = 0.0f;
        for (int h = 0; h < state.n_head[il]; h++) {
            sum += state.importance[il][h];
        }
        if (sum > 0.0f) {
            for (int h = 0; h < state.n_head[il]; h++) {
                state.importance[il][h] /= sum;
                state.importance[il][h] *= state.n_head[il];
            }
        }
    }
}

// Compute head importance using activation analysis or imatrix
static void compute_importance_from_activations(
    llama_nash_state          & state,
    const llama_imatrix_data  * imatrix
) {
    LLAMA_LOG_INFO("%s: computing head importance from activations\n", __func__);

    // If imatrix is provided, use it for accurate importance
    if (imatrix && !imatrix->empty()) {
        if (compute_importance_from_imatrix(state, *imatrix)) {
            return;  // Success
        }
        LLAMA_LOG_WARN("%s: failed to use imatrix, falling back to heuristic\n", __func__);
    }

    // Fallback to heuristic-based importance
    compute_importance_heuristic(state);
}

//
// Redundancy matrix computation
//

static void compute_redundancy_matrix(llama_nash_state & state) {
    LLAMA_LOG_INFO("%s: computing redundancy matrix\n", __func__);

    // Redundancy measures how similar/correlated two heads are
    // For a full implementation, we would:
    // 1. Compute attention patterns for each head on calibration data
    // 2. Measure cosine similarity between patterns
    // 3. Or: measure correlation of output activations

    // For now, use a simple heuristic:
    // - Adjacent heads often learn similar patterns
    // - Heads at similar positions across layers may be redundant

    for (int il = 0; il < state.n_layer; il++) {
        int nh = state.n_head[il];

        for (int hi = 0; hi < nh; hi++) {
            for (int hj = 0; hj < nh; hj++) {
                if (hi == hj) {
                    // Self-redundancy
                    state.redundancy[il][hi][hj] = 1.0f;
                } else {
                    // Redundancy decreases with distance
                    float dist = (float)std::abs(hi - hj);
                    float max_dist = (float)(nh - 1);

                    // Adjacent heads: high redundancy (~0.7)
                    // Distant heads: low redundancy (~0.1)
                    float r = 0.7f * std::exp(-dist / (max_dist * 0.3f)) + 0.1f;

                    state.redundancy[il][hi][hj] = r;
                    state.redundancy[il][hj][hi] = r;  // Symmetric
                }
            }
        }
    }
}

//
// Nash equilibrium solver
//

static void solve_nash_equilibrium(llama_nash_state & state) {
    const float lr     = state.params.learning_rate;
    const float lambda = state.params.lambda;
    const int   n_iter = state.params.n_iterations;
    const float eps    = state.params.convergence_threshold;

    LLAMA_LOG_INFO("%s: solving Nash equilibrium (lambda=%.3f, lr=%.4f, max_iter=%d)\n",
                   __func__, lambda, lr, n_iter);

    // Initialize all participation to 1.0
    for (int il = 0; il < state.n_layer; il++) {
        for (int h = 0; h < state.n_head[il]; h++) {
            state.participation[il][h] = 1.0f;
        }
    }

    // Gradient ascent iterations
    int converged_at = n_iter;
    for (int iter = 0; iter < n_iter; iter++) {
        float max_delta = 0.0f;

        for (int il = 0; il < state.n_layer; il++) {
            int nh = state.n_head[il];

            for (int h = 0; h < nh; h++) {
                // Utility gradient: dU_i/ds_i = c_i - lambda * sum_j(s_j * r_ij)
                float c_i = state.importance[il][h];
                float redundancy_pressure = 0.0f;

                for (int j = 0; j < nh; j++) {
                    redundancy_pressure += state.participation[il][j] *
                                           state.redundancy[il][h][j];
                }

                float gradient = c_i - lambda * redundancy_pressure;

                // Update with projection to [0, 1]
                float old_s = state.participation[il][h];
                float new_s = old_s + lr * gradient;
                new_s = std::max(0.0f, std::min(1.0f, new_s));

                state.participation[il][h] = new_s;
                max_delta = std::max(max_delta, std::abs(new_s - old_s));
            }
        }

        // Check convergence
        if (max_delta < eps) {
            converged_at = iter + 1;
            break;
        }
    }

    LLAMA_LOG_INFO("%s: equilibrium %s at iteration %d (max_delta=%.6f)\n",
                   __func__,
                   converged_at < n_iter ? "converged" : "reached max iterations",
                   converged_at, eps);
}

//
// Apply pruning decisions
//

static llama_nash_result apply_pruning(llama_nash_state & state) {
    llama_nash_result result;

    result.n_head_new.resize(state.n_layer);
    result.n_head_kv_new.resize(state.n_layer);
    result.head_mask.resize(state.n_layer);

    result.total_heads_original = 0;
    result.total_heads_pruned = 0;

    const float threshold = state.params.prune_threshold;
    const float max_prune = state.params.max_prune_ratio;
    const int   min_heads = state.params.min_heads_per_layer;

    for (int il = 0; il < state.n_layer; il++) {
        int nh = state.n_head[il];
        int nh_kv = state.n_head_kv[il];

        result.total_heads_original += nh;
        result.head_mask[il].resize(nh, false);

        // Sort heads by participation score
        std::vector<std::pair<float, int>> sorted;
        for (int h = 0; h < nh; h++) {
            sorted.push_back({state.participation[il][h], h});
        }
        std::sort(sorted.begin(), sorted.end());

        // Determine minimum heads to keep
        int min_keep = std::max(min_heads, (int)std::ceil(nh * (1.0f - max_prune)));

        // For GQA models, ensure n_head remains divisible by n_head_kv
        if (nh_kv > 0 && nh > nh_kv) {
            int gqa_ratio = nh / nh_kv;
            // Round min_keep up to nearest multiple of gqa_ratio
            min_keep = ((min_keep + gqa_ratio - 1) / gqa_ratio) * gqa_ratio;
        }

        // Prune heads below threshold, respecting constraints
        int n_pruned = 0;
        for (const auto & [score, h] : sorted) {
            if (score < threshold && (nh - n_pruned) > min_keep) {
                result.head_mask[il][h] = true;
                n_pruned++;
            }
        }

        // Adjust for GQA divisibility
        if (nh_kv > 0 && nh > nh_kv) {
            int gqa_ratio = nh / nh_kv;
            int kept = nh - n_pruned;
            int remainder = kept % gqa_ratio;
            if (remainder != 0) {
                // Un-prune some heads to maintain divisibility
                int need_unprune = gqa_ratio - remainder;
                for (int i = (int)sorted.size() - 1; i >= 0 && need_unprune > 0; i--) {
                    int h = sorted[i].second;
                    if (result.head_mask[il][h]) {
                        result.head_mask[il][h] = false;
                        n_pruned--;
                        need_unprune--;
                    }
                }
            }
        }

        result.total_heads_pruned += n_pruned;
        result.n_head_new[il] = nh - n_pruned;

        // Adjust KV heads to maintain GQA ratio
        // This is required because attention kernels expect n_head % n_head_kv == 0
        if (nh_kv > 0 && nh > nh_kv && n_pruned > 0) {
            int gqa_ratio = nh / nh_kv;
            int new_kv = (nh - n_pruned) / gqa_ratio;
            if (new_kv < 1) new_kv = 1;  // Keep at least 1 KV head
            result.n_head_kv_new[il] = new_kv;
        } else {
            result.n_head_kv_new[il] = nh_kv;
        }

        // Collect head stats if verbose
        if (state.params.verbose) {
            for (int h = 0; h < nh; h++) {
                llama_head_stats stats;
                stats.layer = il;
                stats.head = h;
                stats.importance = state.importance[il][h];
                stats.participation = state.participation[il][h];
                stats.pruned = result.head_mask[il][h];

                // Compute redundancy sum
                float r_sum = 0.0f;
                for (int j = 0; j < nh; j++) {
                    if (j != h) {
                        r_sum += state.redundancy[il][h][j];
                    }
                }
                stats.redundancy_sum = r_sum;

                result.head_stats.push_back(stats);
            }
        }
    }

    result.total_heads_remaining = result.total_heads_original - result.total_heads_pruned;
    result.prune_ratio = (float)result.total_heads_pruned / (float)result.total_heads_original;

    return result;
}

//
// Public API implementation
//

llama_nash_result llama_nash_compute_pruning(
    const llama_model              & model,
    const llama_nash_prune_params  & params,
    const llama_imatrix_data       * imatrix
) {
    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("%s: starting Nash equilibrium pruning\n", __func__);
    LLAMA_LOG_INFO("%s: model has %d layers\n", __func__, (int)model.hparams.n_layer);

    // Create internal state
    llama_nash_state state(model, params);

    // Step 1: Compute head importance from imatrix or heuristic
    compute_importance_from_activations(state, imatrix);

    // Step 2: Compute redundancy matrix
    compute_redundancy_matrix(state);

    // Step 3: Solve Nash equilibrium
    solve_nash_equilibrium(state);

    // Step 4: Apply pruning decisions
    llama_nash_result result = apply_pruning(state);

    // Log summary
    llama_nash_print_stats(result);

    // Save stats if requested
    if (!params.stats_file.empty()) {
        llama_nash_save_stats(result, params.stats_file);
    }

    return result;
}

bool llama_nash_is_attention_tensor(const std::string & name) {
    // Match attention weight tensors:
    // blk.X.attn_q.weight
    // blk.X.attn_k.weight (not pruned for GQA, but may need adjustment)
    // blk.X.attn_v.weight (not pruned for GQA, but may need adjustment)
    // blk.X.attn_output.weight
    // Also biases and norms

    static const std::regex pattern(
        R"(blk\.(\d+)\.(attn_q|attn_output|attn_q_norm)\.)"
    );

    return std::regex_search(name, pattern);
}

int llama_nash_get_layer_from_name(const std::string & name) {
    static const std::regex pattern(R"(blk\.(\d+)\.)");
    std::smatch match;

    if (std::regex_search(name, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1;
}

ggml_tensor * llama_nash_slice_tensor(
    ggml_context              * ctx,
    ggml_tensor               * tensor,
    const llama_nash_result   & result,
    const std::string         & name,
    int                         n_embd,
    int                         n_embd_head_k
) {
    (void)n_embd;  // May be used in future for validation

    int layer = llama_nash_get_layer_from_name(name);
    if (layer < 0 || layer >= (int)result.head_mask.size()) {
        return tensor;  // Not a layer tensor, return unchanged
    }

    const auto & mask = result.head_mask[layer];
    int n_heads_orig = (int)mask.size();
    int n_heads_new  = result.n_heads_kept(layer);

    if (n_heads_new == n_heads_orig) {
        return tensor;  // No pruning for this layer
    }

    // Determine if this is a Q tensor or output tensor
    bool is_q      = name.find("attn_q") != std::string::npos && name.find("attn_output") == std::string::npos;
    bool is_output = name.find("attn_output") != std::string::npos;
    bool is_bias   = name.find("bias") != std::string::npos;
    bool is_norm   = name.find("norm") != std::string::npos;

    if (!is_q && !is_output) {
        return tensor;  // K, V tensors handled differently for GQA
    }

    LLAMA_LOG_DEBUG("%s: slicing %s: %d -> %d heads\n",
                    __func__, name.c_str(), n_heads_orig, n_heads_new);

    // Calculate dimensions
    int64_t head_dim = n_embd_head_k;

    ggml_tensor * sliced = nullptr;

    if (is_bias || is_norm) {
        // Bias/norm: 1D tensor [head_dim * n_heads]
        int64_t new_size = head_dim * n_heads_new;
        sliced = ggml_new_tensor_1d(ctx, tensor->type, new_size);

        // Copy kept head slices
        char * dst = (char *)sliced->data;
        char * src = (char *)tensor->data;
        size_t head_bytes = head_dim * ggml_type_size(tensor->type);

        int dst_head = 0;
        for (int h = 0; h < n_heads_orig; h++) {
            if (!mask[h]) {
                memcpy(dst + dst_head * head_bytes,
                       src + h * head_bytes,
                       head_bytes);
                dst_head++;
            }
        }
    } else if (is_q) {
        // wq: [n_embd, head_dim * n_heads] -> [n_embd, head_dim * n_heads_new]
        int64_t ne0 = tensor->ne[0];  // n_embd
        int64_t ne1_new = head_dim * n_heads_new;

        sliced = ggml_new_tensor_2d(ctx, tensor->type, ne0, ne1_new);

        // Copy column slices for kept heads
        size_t col_bytes = ne0 * ggml_type_size(tensor->type);
        char * dst = (char *)sliced->data;
        char * src = (char *)tensor->data;

        int dst_head = 0;
        for (int h = 0; h < n_heads_orig; h++) {
            if (!mask[h]) {
                for (int d = 0; d < (int)head_dim; d++) {
                    int src_col = h * head_dim + d;
                    int dst_col = dst_head * head_dim + d;
                    memcpy(dst + dst_col * col_bytes,
                           src + src_col * col_bytes,
                           col_bytes);
                }
                dst_head++;
            }
        }
    } else if (is_output) {
        // wo: [head_dim * n_heads, n_embd] -> [head_dim * n_heads_new, n_embd]
        int64_t ne0_new = head_dim * n_heads_new;
        int64_t ne1 = tensor->ne[1];  // n_embd

        sliced = ggml_new_tensor_2d(ctx, tensor->type, ne0_new, ne1);

        // Copy row slices for kept heads
        size_t elem_size = ggml_type_size(tensor->type);
        char * dst = (char *)sliced->data;
        char * src = (char *)tensor->data;

        for (int64_t col = 0; col < ne1; col++) {
            int dst_head = 0;
            for (int h = 0; h < n_heads_orig; h++) {
                if (!mask[h]) {
                    for (int d = 0; d < (int)head_dim; d++) {
                        int src_row = h * head_dim + d;
                        int dst_row = dst_head * head_dim + d;
                        memcpy(dst + (col * ne0_new + dst_row) * elem_size,
                               src + (col * tensor->ne[0] + src_row) * elem_size,
                               elem_size);
                    }
                    dst_head++;
                }
            }
        }
    }

    return sliced ? sliced : tensor;
}

void llama_nash_print_stats(const llama_nash_result & result) {
    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("=== Nash Equilibrium Pruning Results ===\n");
    LLAMA_LOG_INFO("  Total heads original:  %d\n", result.total_heads_original);
    LLAMA_LOG_INFO("  Total heads pruned:    %d\n", result.total_heads_pruned);
    LLAMA_LOG_INFO("  Total heads remaining: %d\n", result.total_heads_remaining);
    LLAMA_LOG_INFO("  Prune ratio:           %.1f%%\n", result.prune_ratio * 100.0f);
    LLAMA_LOG_INFO("\n");

    // Per-layer summary
    LLAMA_LOG_INFO("  Per-layer head counts:\n");
    for (size_t il = 0; il < result.n_head_new.size(); il++) {
        int orig = (int)result.head_mask[il].size();
        int kept = (int)result.n_head_new[il];
        int pruned = orig - kept;

        if (pruned > 0) {
            LLAMA_LOG_INFO("    Layer %3zu: %2d -> %2d heads (-%d)\n",
                           il, orig, kept, pruned);
        }
    }
    LLAMA_LOG_INFO("\n");
}

bool llama_nash_save_stats(const llama_nash_result & result, const std::string & path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        LLAMA_LOG_ERROR("%s: failed to open %s for writing\n", __func__, path.c_str());
        return false;
    }

    f << "{\n";
    f << "  \"summary\": {\n";
    f << "    \"total_heads_original\": " << result.total_heads_original << ",\n";
    f << "    \"total_heads_pruned\": " << result.total_heads_pruned << ",\n";
    f << "    \"total_heads_remaining\": " << result.total_heads_remaining << ",\n";
    f << "    \"prune_ratio\": " << result.prune_ratio << "\n";
    f << "  },\n";

    f << "  \"layers\": [\n";
    for (size_t il = 0; il < result.n_head_new.size(); il++) {
        f << "    {";
        f << "\"layer\": " << il << ", ";
        f << "\"n_head_orig\": " << result.head_mask[il].size() << ", ";
        f << "\"n_head_new\": " << result.n_head_new[il] << ", ";
        f << "\"n_head_kv_new\": " << result.n_head_kv_new[il];
        f << "}" << (il < result.n_head_new.size() - 1 ? "," : "") << "\n";
    }
    f << "  ]";

    if (!result.head_stats.empty()) {
        f << ",\n  \"heads\": [\n";
        for (size_t i = 0; i < result.head_stats.size(); i++) {
            const auto & s = result.head_stats[i];
            f << "    {";
            f << "\"layer\": " << s.layer << ", ";
            f << "\"head\": " << s.head << ", ";
            f << "\"importance\": " << s.importance << ", ";
            f << "\"participation\": " << s.participation << ", ";
            f << "\"redundancy_sum\": " << s.redundancy_sum << ", ";
            f << "\"pruned\": " << (s.pruned ? "true" : "false");
            f << "}" << (i < result.head_stats.size() - 1 ? "," : "") << "\n";
        }
        f << "  ]";
    }

    f << "\n}\n";

    LLAMA_LOG_INFO("%s: saved stats to %s\n", __func__, path.c_str());
    return true;
}

bool llama_nash_get_pruned_dims(
    const llama_nash_result & result,
    const std::string       & name,
    int64_t                   ne[4],
    int64_t                   new_ne[4],
    int                       n_embd,
    int                       n_embd_head_k
) {
    (void)n_embd;  // May be used in future

    // Copy original dimensions
    for (int i = 0; i < 4; i++) {
        new_ne[i] = ne[i];
    }

    int layer = llama_nash_get_layer_from_name(name);
    if (layer < 0 || layer >= (int)result.head_mask.size()) {
        return false;
    }

    const auto & mask = result.head_mask[layer];
    int n_heads_orig = (int)mask.size();
    int n_heads_new = result.n_heads_kept(layer);

    if (n_heads_new == n_heads_orig || n_heads_new == 0) {
        return false;  // No pruning for this layer
    }

    bool is_q      = name.find("attn_q") != std::string::npos && name.find("attn_output") == std::string::npos;
    bool is_k      = name.find("attn_k") != std::string::npos;
    bool is_v      = name.find("attn_v") != std::string::npos;
    bool is_output = name.find("attn_output") != std::string::npos;
    bool is_bias   = name.find(".bias") != std::string::npos;

    // Get KV head reduction for this layer
    int n_heads_kv_new = (int)result.n_head_kv_new[layer];
    int n_heads_kv_orig = (int)(result.n_head_new[layer] > 0 ?
                                (n_heads_orig * result.n_head_kv_new[layer]) / result.n_head_new[layer] :
                                result.n_head_kv_new[layer]);
    // Approximate original KV heads from the GQA ratio
    int gqa_ratio = (result.n_head_new[layer] > 0 && result.n_head_kv_new[layer] > 0) ?
                    result.n_head_new[layer] / result.n_head_kv_new[layer] : 1;
    if (gqa_ratio < 1) gqa_ratio = 1;
    n_heads_kv_orig = n_heads_orig / gqa_ratio;
    if (n_heads_kv_orig < 1) n_heads_kv_orig = 1;

    if (!is_q && !is_k && !is_v && !is_output) {
        return false;
    }

    int64_t head_dim = n_embd_head_k;

    if (is_bias) {
        if (is_q) {
            // Q bias: 1D tensor [head_dim * n_heads] -> [head_dim * n_heads_new]
            new_ne[0] = head_dim * n_heads_new;
        } else if (is_k || is_v) {
            // K/V bias: [head_dim * n_heads_kv] -> [head_dim * n_heads_kv_new]
            if (n_heads_kv_new != n_heads_kv_orig) {
                new_ne[0] = head_dim * n_heads_kv_new;
            } else {
                return false;
            }
        }
    } else if (is_q) {
        // wq: [n_embd, head_dim * n_heads] -> [n_embd, head_dim * n_heads_new]
        new_ne[1] = head_dim * n_heads_new;
    } else if (is_k || is_v) {
        // wk/wv: [n_embd, head_dim * n_heads_kv] -> [n_embd, head_dim * n_heads_kv_new]
        if (n_heads_kv_new != n_heads_kv_orig) {
            new_ne[1] = head_dim * n_heads_kv_new;
        } else {
            return false;
        }
    } else if (is_output) {
        // wo: [head_dim * n_heads, n_embd] -> [head_dim * n_heads_new, n_embd]
        new_ne[0] = head_dim * n_heads_new;
    }

    return true;
}

size_t llama_nash_slice_data(
    const llama_nash_result & result,
    const std::string       & name,
    const void              * data_in,
    void                    * data_out,
    int64_t                   ne[4],
    size_t                    type_size,
    int                       n_embd,
    int                       n_embd_head_k
) {
    (void)n_embd;  // May be used in future for validation

    int layer = llama_nash_get_layer_from_name(name);
    if (layer < 0 || layer >= (int)result.head_mask.size()) {
        // Just copy the data unchanged
        size_t total_size = ne[0] * ne[1] * ne[2] * ne[3] * type_size;
        memcpy(data_out, data_in, total_size);
        return total_size;
    }

    const auto & mask = result.head_mask[layer];
    int n_heads_orig = (int)mask.size();
    int n_heads_new = result.n_heads_kept(layer);

    if (n_heads_new == n_heads_orig || n_heads_new == 0) {
        size_t total_size = ne[0] * ne[1] * ne[2] * ne[3] * type_size;
        memcpy(data_out, data_in, total_size);
        return total_size;
    }

    bool is_q      = name.find("attn_q") != std::string::npos && name.find("attn_output") == std::string::npos;
    bool is_k      = name.find("attn_k") != std::string::npos;
    bool is_v      = name.find("attn_v") != std::string::npos;
    bool is_output = name.find("attn_output") != std::string::npos;
    bool is_bias   = name.find(".bias") != std::string::npos;

    // Calculate KV head counts
    int gqa_ratio = (result.n_head_new[layer] > 0 && result.n_head_kv_new[layer] > 0) ?
                    result.n_head_new[layer] / result.n_head_kv_new[layer] : 1;
    if (gqa_ratio < 1) gqa_ratio = 1;
    int n_heads_kv_orig = n_heads_orig / gqa_ratio;
    if (n_heads_kv_orig < 1) n_heads_kv_orig = 1;
    int n_heads_kv_new = (int)result.n_head_kv_new[layer];

    if (!is_q && !is_k && !is_v && !is_output) {
        size_t total_size = ne[0] * ne[1] * ne[2] * ne[3] * type_size;
        memcpy(data_out, data_in, total_size);
        return total_size;
    }

    // Check if KV tensors actually need slicing
    if ((is_k || is_v) && n_heads_kv_new == n_heads_kv_orig) {
        size_t total_size = ne[0] * ne[1] * ne[2] * ne[3] * type_size;
        memcpy(data_out, data_in, total_size);
        return total_size;
    }

    int64_t head_dim = n_embd_head_k;
    const char * src = (const char *)data_in;
    char * dst = (char *)data_out;

    LLAMA_LOG_DEBUG("%s: slicing %s layer %d: %d -> %d heads\n",
                    __func__, name.c_str(), layer, n_heads_orig, n_heads_new);

    if (is_bias) {
        if (is_q) {
            // Q Bias: 1D [head_dim * n_heads]
            size_t head_bytes = head_dim * type_size;
            size_t dst_offset = 0;

            for (int h = 0; h < n_heads_orig; h++) {
                if (!mask[h]) {
                    memcpy(dst + dst_offset, src + h * head_bytes, head_bytes);
                    dst_offset += head_bytes;
                }
            }
            return dst_offset;
        } else if (is_k || is_v) {
            // K/V Bias: Keep first n_heads_kv_new heads
            size_t copy_size = head_dim * n_heads_kv_new * type_size;
            memcpy(dst, src, copy_size);
            return copy_size;
        }
    } else if (is_k || is_v) {
        // wk/wv: [n_embd, head_dim * n_heads_kv] - keep first n_heads_kv_new heads
        int64_t n_rows = ne[0];
        size_t old_cols = ne[1];
        size_t new_cols = head_dim * n_heads_kv_new;

        // For each row, copy only the columns for kept heads
        for (int64_t row = 0; row < n_rows; row++) {
            memcpy(dst + row * new_cols * type_size,
                   src + row * old_cols * type_size,
                   new_cols * type_size);
        }
        return n_rows * new_cols * type_size;
    } else if (is_q) {
        // wq: [n_embd, head_dim * n_heads] - columns are heads
        // ne[0] = n_embd (rows), ne[1] = head_dim * n_heads (cols)
        int64_t n_rows = ne[0];
        size_t row_bytes = type_size;  // Per element
        size_t new_cols = head_dim * n_heads_new;

        for (int64_t row = 0; row < n_rows; row++) {
            size_t dst_col = 0;
            for (int h = 0; h < n_heads_orig; h++) {
                if (!mask[h]) {
                    for (int64_t d = 0; d < head_dim; d++) {
                        int64_t src_col = h * head_dim + d;
                        memcpy(dst + (row * new_cols + dst_col) * row_bytes,
                               src + (row * ne[1] + src_col) * row_bytes,
                               row_bytes);
                        dst_col++;
                    }
                }
            }
        }
        return n_rows * new_cols * type_size;

    } else if (is_output) {
        // wo: [head_dim * n_heads, n_embd] - rows are heads
        // ne[0] = head_dim * n_heads (rows), ne[1] = n_embd (cols)
        int64_t n_cols = ne[1];
        size_t elem_bytes = type_size;
        int64_t new_rows = head_dim * n_heads_new;

        for (int64_t col = 0; col < n_cols; col++) {
            size_t dst_row = 0;
            for (int h = 0; h < n_heads_orig; h++) {
                if (!mask[h]) {
                    for (int64_t d = 0; d < head_dim; d++) {
                        int64_t src_row = h * head_dim + d;
                        memcpy(dst + (col * new_rows + dst_row) * elem_bytes,
                               src + (col * ne[0] + src_row) * elem_bytes,
                               elem_bytes);
                        dst_row++;
                    }
                }
            }
        }
        return new_rows * n_cols * type_size;
    }

    // Fallback: copy unchanged
    size_t total_size = ne[0] * ne[1] * ne[2] * ne[3] * type_size;
    memcpy(data_out, data_in, total_size);
    return total_size;
}
