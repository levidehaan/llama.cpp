#pragma once

#include "llama.h"

#include <string>
#include <vector>

//
// Nash Equilibrium-Based Attention Head Pruning
//
// This module implements game-theoretic pruning of attention heads using Nash equilibrium.
// Each attention head is treated as a "player" with a participation variable s_i in [0,1].
// The equilibrium solution identifies redundant heads that can be removed with minimal
// impact on model quality.
//
// Reference: docs/nash-equilibrium-pruning.md
//

// Pruning parameters for Nash equilibrium computation
struct llama_nash_prune_params {
    // === Nash Equilibrium Parameters ===
    float lambda               = 0.5f;    // Redundancy penalty weight [0.1-1.0]
    float learning_rate        = 0.01f;   // Gradient ascent step size
    int   n_iterations         = 100;     // Max equilibrium iterations
    float convergence_threshold = 1e-4f;  // Stop when max delta_s < threshold

    // === Pruning Thresholds ===
    float prune_threshold      = 0.1f;    // Heads with s_i < threshold are pruned
    int   min_heads_per_layer  = 1;       // Never prune below this count
    float max_prune_ratio      = 0.7f;    // Max fraction to prune per layer

    // === Calibration ===
    std::string calibration_file;         // Path to calibration text file (optional)
    int   n_calibration_tokens = 512;     // Number of tokens to use
    int   n_calibration_batches = 4;      // Batches for averaging

    // === Output ===
    bool  verbose              = false;   // Print per-head statistics
    std::string stats_file;               // Save detailed stats to JSON (optional)
};

// Per-head statistics for analysis and debugging
struct llama_head_stats {
    int   layer;
    int   head;
    float importance;        // Raw contribution score (c_i)
    float participation;     // Nash equilibrium value (s_i)
    float redundancy_sum;    // Total redundancy with other heads
    bool  pruned;
};

// Result of Nash pruning computation
struct llama_nash_result {
    // Per-layer new head counts after pruning
    std::vector<uint32_t> n_head_new;
    std::vector<uint32_t> n_head_kv_new;

    // Per-layer pruning masks [layer][head] - true means pruned
    std::vector<std::vector<bool>> head_mask;

    // Statistics
    int total_heads_original;
    int total_heads_pruned;
    int total_heads_remaining;
    float prune_ratio;

    // Detailed per-head stats (if verbose)
    std::vector<llama_head_stats> head_stats;

    // Check if layer/head is pruned
    bool is_pruned(int layer, int head) const {
        if (layer < 0 || layer >= (int)head_mask.size()) return false;
        if (head < 0 || head >= (int)head_mask[layer].size()) return false;
        return head_mask[layer][head];
    }

    // Get number of kept heads for a layer
    int n_heads_kept(int layer) const {
        if (layer < 0 || layer >= (int)n_head_new.size()) return 0;
        return (int)n_head_new[layer];
    }
};

// Forward declarations
struct llama_model;
struct llama_context;
struct ggml_context;
struct ggml_tensor;

//
// Main API
//

// Compute Nash equilibrium pruning decisions for a model
// Uses calibration data or imatrix for importance estimation
// Returns pruning result with per-layer head masks
llama_nash_result llama_nash_compute_pruning(
    const llama_model              & model,
    const llama_nash_prune_params  & params,
    const std::vector<float>       * imatrix_data = nullptr  // Optional: use imatrix for importance
);

// Check if a tensor name corresponds to an attention weight that should be pruned
bool llama_nash_is_attention_tensor(const std::string & name);

// Slice an attention tensor to remove pruned heads
// Returns a new tensor with reduced dimensions
ggml_tensor * llama_nash_slice_tensor(
    ggml_context              * ctx,
    ggml_tensor               * tensor,
    const llama_nash_result   & result,
    const std::string         & name,
    int                         n_embd,
    int                         n_embd_head_k
);

// Get the layer index from a tensor name (e.g., "blk.5.attn_q.weight" -> 5)
// Returns -1 if not a layer tensor
int llama_nash_get_layer_from_name(const std::string & name);

// Calculate new dimensions for a pruned tensor
// Returns true if tensor should be sliced, false otherwise
// Updates new_ne with the new dimensions
bool llama_nash_get_pruned_dims(
    const llama_nash_result & result,
    const std::string       & name,
    int64_t                   ne[4],      // Original dimensions
    int64_t                   new_ne[4],  // Output: new dimensions
    int                       n_embd,
    int                       n_embd_head_k
);

// Slice tensor data in-place, returns the new size in bytes
// data_out must be pre-allocated with sufficient size
size_t llama_nash_slice_data(
    const llama_nash_result & result,
    const std::string       & name,
    const void              * data_in,
    void                    * data_out,
    int64_t                   ne[4],
    size_t                    type_size,
    int                       n_embd,
    int                       n_embd_head_k
);

// Print pruning statistics to log
void llama_nash_print_stats(const llama_nash_result & result);

// Save detailed statistics to JSON file
bool llama_nash_save_stats(const llama_nash_result & result, const std::string & path);
