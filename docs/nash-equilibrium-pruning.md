# Nash Equilibrium-Based Attention Head Pruning for llama.cpp

> **Status**: Experimental
> **Branch**: `claude/add-theorizing-llama-cpp-PMdRy`
> **Author**: Experimental implementation based on game-theoretic pruning research

## Executive Summary

This document describes the implementation of Nash equilibrium-based attention head pruning in llama.cpp. The approach treats attention heads as rational agents competing for participation, using game theory to identify and remove redundant heads while preserving model quality.

**Expected Results:**
- 50-70% attention head reduction
- ~20-30% additional memory savings on top of quantization
- Minimal perplexity impact when properly tuned
- No changes needed to inference code (llama-cli, llama-server work automatically)

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Structures](#3-data-structures)
4. [Algorithm Details](#4-algorithm-details)
5. [GGUF Format Extensions](#5-gguf-format-extensions)
6. [Integration Points](#6-integration-points)
7. [CLI Usage](#7-cli-usage)
8. [Testing Strategy](#8-testing-strategy)
9. [Memory Savings Analysis](#9-memory-savings-analysis)
10. [Implementation Phases](#10-implementation-phases)
11. [File Reference](#11-file-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Theoretical Background

### 1.1 The Problem

Large Language Models have significant redundancy in their attention heads. Many heads learn similar patterns or contribute minimally to output quality. Traditional pruning methods use heuristics (magnitude, gradient) that don't account for inter-head relationships.

### 1.2 Nash Equilibrium Approach

We model attention heads as players in a non-cooperative game:

- **Players**: Each attention head `i` in layer `l`
- **Strategy**: Participation level `s_i ∈ [0, 1]` (1 = fully active, 0 = pruned)
- **Utility Function**: Balances contribution against redundancy

```
U_i(s) = s_i × c_i - λ × s_i × Σ_j(s_j × r_ij)

Where:
  c_i = contribution score (how much head i helps the loss)
  r_ij = redundancy score (correlation between heads i and j)
  λ = redundancy penalty weight (hyperparameter)
```

### 1.3 Nash Equilibrium Condition

At equilibrium, no head can improve its utility by unilaterally changing participation:

```
∂U_i/∂s_i = c_i - λ × Σ_j(s_j × r_ij) = 0  (for interior solutions)
```

Heads with low equilibrium `s_i` values are redundant and can be pruned.

### 1.4 Why This Works

- **Contribution-aware**: Keeps heads that actually help model performance
- **Redundancy-aware**: Removes heads that duplicate others' work
- **Stable**: Nash equilibrium ensures pruning decisions are mutually consistent
- **Principled**: Game theory provides theoretical guarantees

---

## 2. Architecture Overview

### 2.1 High-Level Flow

```
                                    ┌────────────────────────────┐
                                    │     llama-quantize CLI     │
                                    │  --nash-prune <options>    │
                                    └─────────────┬──────────────┘
                                                  │
                                                  ▼
┌──────────────────┐    ┌─────────────────────────────────────────────────┐
│  Input Model     │    │              Nash Pruning Pipeline              │
│  (F16/F32 GGUF)  │───▶│                                                 │
└──────────────────┘    │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
                        │  │ Compute │  │  Nash   │  │  Prune  │         │
┌──────────────────┐    │  │  Head   │─▶│ Equil.  │─▶│ Tensors │         │
│  Calibration     │───▶│  │ Import. │  │ Solver  │  │         │         │
│  Data (optional) │    │  └─────────┘  └─────────┘  └─────────┘         │
└──────────────────┘    │                                                 │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │            Standard Quantization                │
                        │         (Q4_K_M, Q5_K_M, etc.)                 │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │              Output Model                       │
                        │     (Pruned + Quantized GGUF)                  │
                        └─────────────────────────────────────────────────┘
```

### 2.2 Key Insight: Existing Infrastructure

llama.cpp already supports per-layer head counts via:
- `llama_hparams::n_head_arr[LLAMA_MAX_LAYERS]`
- `llama_hparams::n_head_kv_arr[LLAMA_MAX_LAYERS]`

This means pruned models work automatically in llama-cli and llama-server!

### 2.3 Structured vs Unstructured Pruning

We use **structured pruning** (removing entire heads) rather than unstructured pruning (zeroing individual weights):

| Aspect | Structured | Unstructured |
|--------|------------|--------------|
| Memory savings | Real reduction | Only with sparse kernels |
| Speed improvement | Automatic | Requires sparse ops |
| Implementation | Tensor slicing | Weight masking |
| Compatibility | Full | Limited |

---

## 3. Data Structures

### 3.1 Pruning Parameters

```cpp
// Location: src/llama-prune.h

struct llama_nash_prune_params {
    // === Nash Equilibrium Parameters ===
    float lambda;                    // Redundancy penalty weight [0.1-1.0], default: 0.5
    float learning_rate;             // Gradient ascent step size, default: 0.01
    int   n_iterations;              // Max equilibrium iterations, default: 100
    float convergence_threshold;     // Stop when max Δs_i < threshold, default: 1e-4

    // === Pruning Thresholds ===
    float prune_threshold;           // Heads with s_i < threshold are pruned, default: 0.1
    int   min_heads_per_layer;       // Never prune below this count, default: 1
    float max_prune_ratio;           // Max fraction to prune per layer, default: 0.7

    // === Calibration ===
    std::string calibration_file;    // Path to calibration text file
    int   n_calibration_tokens;      // Number of tokens to use, default: 512
    int   n_calibration_batches;     // Number of batches for averaging, default: 4

    // === Output ===
    bool  verbose;                   // Print per-head statistics
    std::string stats_file;          // Save detailed stats to JSON
};
```

### 3.2 Internal State

```cpp
// Location: src/llama-prune.cpp

struct llama_nash_state {
    const llama_model & model;
    const llama_nash_prune_params & params;

    int n_layer;
    std::vector<int> n_head;           // Original head counts per layer
    std::vector<int> n_head_kv;        // Original KV head counts per layer

    // === Per-Head Data [layer][head] ===
    std::vector<std::vector<float>> importance;     // c_i: contribution scores
    std::vector<std::vector<float>> participation;  // s_i: Nash equilibrium values
    std::vector<std::vector<bool>>  pruned;         // Final pruning decisions

    // === Redundancy Matrix [layer][head_i][head_j] ===
    std::vector<std::vector<std::vector<float>>> redundancy;  // r_ij

    // === Results ===
    std::vector<int> new_n_head;       // Pruned head counts per layer
    std::vector<int> new_n_head_kv;    // Pruned KV head counts (for GQA consistency)

    int total_heads_original;
    int total_heads_pruned;
    int total_heads_remaining;
};
```

### 3.3 Head Statistics (for debugging/analysis)

```cpp
struct llama_head_stats {
    int   layer;
    int   head;
    float importance;        // Raw contribution score
    float participation;     // Nash equilibrium s_i value
    float redundancy_sum;    // Total redundancy with other heads
    bool  pruned;

    // Optional: detailed metrics
    float attention_entropy; // How focused the head's attention is
    float activation_norm;   // Average activation magnitude
};
```

---

## 4. Algorithm Details

### 4.1 Step 1: Compute Head Importance

For each attention head, measure how much it contributes to model output:

```cpp
void compute_head_importance(llama_nash_state & state, llama_context * ctx) {
    // Method A: Activation-based (fast, approximate)
    // Run forward pass, measure ||head_output||_F for each head

    // Method B: Gradient-based (accurate, slower)
    // Compute gradient of loss w.r.t. head outputs

    // Method C: Ablation-based (most accurate, slowest)
    // Measure perplexity change when zeroing each head

    for (int il = 0; il < n_layer; il++) {
        for (int h = 0; h < n_head[il]; h++) {
            // Normalize importance scores within each layer
            state.importance[il][h] = head_activation_norm[il][h] / layer_total[il];
        }
    }
}
```

**Approximation using imatrix:**
If an importance matrix file exists (from `llama-imatrix`), we can approximate head importance from the attention weight importance values:

```cpp
void compute_importance_from_imatrix(llama_nash_state & state,
                                      const std::vector<float> & imatrix) {
    // Extract importance for wq tensors, aggregate by head
    for (int il = 0; il < n_layer; il++) {
        for (int h = 0; h < n_head[il]; h++) {
            // Sum imatrix values for this head's slice of wq
            int offset = il * n_embd * n_head[il] * head_dim + h * head_dim;
            state.importance[il][h] = sum(imatrix[offset : offset + head_dim]);
        }
    }
}
```

### 4.2 Step 2: Compute Redundancy Matrix

Measure correlation/similarity between heads:

```cpp
void compute_redundancy_matrix(llama_nash_state & state,
                                const std::vector<ggml_tensor*> & activations) {
    for (int il = 0; il < n_layer; il++) {
        for (int hi = 0; hi < n_head[il]; hi++) {
            for (int hj = hi; hj < n_head[il]; hj++) {
                if (hi == hj) {
                    state.redundancy[il][hi][hj] = 1.0f;  // Self-redundancy
                } else {
                    // Cosine similarity of attention patterns or output activations
                    float sim = cosine_similarity(
                        head_activations[il][hi],
                        head_activations[il][hj]
                    );
                    state.redundancy[il][hi][hj] = sim;
                    state.redundancy[il][hj][hi] = sim;  // Symmetric
                }
            }
        }
    }
}
```

### 4.3 Step 3: Nash Equilibrium Solver

Iterative gradient ascent to find equilibrium participation values:

```cpp
void solve_nash_equilibrium(llama_nash_state & state) {
    const float lr = state.params.learning_rate;
    const float lambda = state.params.lambda;

    // Initialize all participation to 1.0 (all heads active)
    for (int il = 0; il < n_layer; il++) {
        for (int h = 0; h < n_head[il]; h++) {
            state.participation[il][h] = 1.0f;
        }
    }

    // Gradient ascent iterations
    for (int iter = 0; iter < state.params.n_iterations; iter++) {
        float max_delta = 0.0f;

        for (int il = 0; il < n_layer; il++) {
            for (int h = 0; h < n_head[il]; h++) {
                // Compute gradient: ∂U_i/∂s_i = c_i - λ × Σ_j(s_j × r_ij)
                float c_i = state.importance[il][h];
                float redundancy_pressure = 0.0f;

                for (int j = 0; j < n_head[il]; j++) {
                    redundancy_pressure += state.participation[il][j] *
                                           state.redundancy[il][h][j];
                }

                float gradient = c_i - lambda * redundancy_pressure;

                // Update with projection to [0, 1]
                float old_s = state.participation[il][h];
                float new_s = std::clamp(old_s + lr * gradient, 0.0f, 1.0f);
                state.participation[il][h] = new_s;

                max_delta = std::max(max_delta, std::abs(new_s - old_s));
            }
        }

        // Check convergence
        if (max_delta < state.params.convergence_threshold) {
            LLAMA_LOG_INFO("Nash equilibrium converged at iteration %d\n", iter);
            break;
        }
    }
}
```

### 4.4 Step 4: Apply Pruning Decisions

```cpp
void apply_pruning(llama_nash_state & state) {
    for (int il = 0; il < n_layer; il++) {
        // Sort heads by participation score
        std::vector<std::pair<float, int>> sorted;
        for (int h = 0; h < n_head[il]; h++) {
            sorted.push_back({state.participation[il][h], h});
        }
        std::sort(sorted.begin(), sorted.end());

        // Determine how many to prune (respecting constraints)
        int n_total = n_head[il];
        int min_keep = std::max(state.params.min_heads_per_layer,
                                (int)std::ceil(n_total * (1.0f - state.params.max_prune_ratio)));

        int n_pruned = 0;
        for (auto & [score, h] : sorted) {
            if (score < state.params.prune_threshold &&
                (n_total - n_pruned) > min_keep) {
                state.pruned[il][h] = true;
                n_pruned++;
            }
        }

        state.new_n_head[il] = n_total - n_pruned;
        state.total_heads_pruned += n_pruned;
    }

    state.total_heads_remaining = state.total_heads_original - state.total_heads_pruned;
}
```

### 4.5 Step 5: Slice Attention Tensors

Remove pruned head dimensions from weight tensors:

```cpp
ggml_tensor * slice_attention_tensor(
    ggml_context * ctx,
    ggml_tensor * tensor,           // Original tensor
    const std::vector<bool> & mask, // true = pruned
    int head_dim,
    bool is_output                  // true for wo, false for wq
) {
    int n_heads_orig = mask.size();
    int n_heads_new = std::count(mask.begin(), mask.end(), false);

    // Create new tensor with reduced dimensions
    int64_t ne0, ne1;
    if (is_output) {
        // wo: [head_dim * n_heads, n_embd] -> [head_dim * n_heads_new, n_embd]
        ne0 = head_dim * n_heads_new;
        ne1 = tensor->ne[1];
    } else {
        // wq: [n_embd, head_dim * n_heads] -> [n_embd, head_dim * n_heads_new]
        ne0 = tensor->ne[0];
        ne1 = head_dim * n_heads_new;
    }

    ggml_tensor * sliced = ggml_new_tensor_2d(ctx, tensor->type, ne0, ne1);

    // Copy only kept heads
    // ... (implementation depends on tensor layout and type)

    return sliced;
}
```

---

## 5. GGUF Format Extensions

### 5.1 Existing Support (No Changes Needed)

The GGUF format already supports per-layer head counts:

```
llama.attention.head_count     = [32, 32, 30, 28, ...]  # Can be array
llama.attention.head_count_kv  = [8, 8, 8, 7, ...]      # Can be array
```

### 5.2 New Metadata Keys (Optional)

For tracking and reproducibility:

```
llama.pruning.method              = "nash_equilibrium"
llama.pruning.version             = 1
llama.pruning.original_head_count = 32
llama.pruning.total_heads_pruned  = 156
llama.pruning.prune_ratio         = 0.48
llama.pruning.lambda              = 0.5
llama.pruning.threshold           = 0.1
llama.pruning.calibration_tokens  = 2048
```

### 5.3 Tensor Shape Changes

For a layer with 32 heads pruned to 20 heads (head_dim=128, n_embd=4096):

| Tensor | Original Shape | Pruned Shape | Savings |
|--------|---------------|--------------|---------|
| `blk.X.attn_q.weight` | [4096, 4096] | [4096, 2560] | 37.5% |
| `blk.X.attn_k.weight` | [4096, 1024] | unchanged | 0% |
| `blk.X.attn_v.weight` | [4096, 1024] | unchanged | 0% |
| `blk.X.attn_output.weight` | [4096, 4096] | [2560, 4096] | 37.5% |

---

## 6. Integration Points

### 6.1 API Extension (include/llama.h)

```cpp
// Add to llama_model_quantize_params struct:

struct llama_model_quantize_params {
    // ... existing fields ...

    // Nash pruning (set to non-null to enable)
    void * nash_prune_params;        // llama_nash_prune_params *
};

// New public API functions:

// Prune model without quantization
LLAMA_API int llama_model_prune(
    const char * fname_inp,
    const char * fname_out,
    const llama_nash_prune_params * params
);

// Get pruning statistics from a model
LLAMA_API bool llama_model_is_pruned(const llama_model * model);
LLAMA_API int  llama_model_heads_pruned(const llama_model * model);
```

### 6.2 Quantization Pipeline (src/llama-quant.cpp)

Insert pruning before quantization:

```cpp
static void llama_model_quantize_impl(...) {
    // ... existing setup ...

    // NEW: Nash pruning phase
    std::unique_ptr<llama_nash_state> prune_state;

    if (params->nash_prune_params) {
        auto * nash_params = (llama_nash_prune_params *)params->nash_prune_params;

        LLAMA_LOG_INFO("Running Nash equilibrium pruning...\n");

        prune_state = std::make_unique<llama_nash_state>(model, *nash_params);

        // Step 1: Compute importance
        compute_head_importance(*prune_state, ctx);

        // Step 2: Compute redundancy
        compute_redundancy_matrix(*prune_state);

        // Step 3: Solve Nash equilibrium
        solve_nash_equilibrium(*prune_state);

        // Step 4: Apply pruning decisions
        apply_pruning(*prune_state);

        LLAMA_LOG_INFO("Pruning: %d/%d heads removed (%.1f%%)\n",
            prune_state->total_heads_pruned,
            prune_state->total_heads_original,
            100.0f * prune_state->total_heads_pruned / prune_state->total_heads_original);

        // Update GGUF metadata with new head counts
        write_pruned_metadata(ctx_out, *prune_state);
    }

    // ... existing tensor loop ...

    for (const auto * tensor : tensors) {
        // NEW: Apply tensor slicing if pruning enabled
        if (prune_state && is_attention_tensor(tensor->name)) {
            tensor = slice_attention_tensor(ctx, tensor, *prune_state);
        }

        // ... rest of quantization ...
    }
}
```

---

## 7. CLI Usage

### 7.1 Basic Usage

```bash
# Prune + Quantize in one step
./llama-quantize --nash-prune \
    model-f16.gguf \
    model-pruned-Q4_K_M.gguf \
    Q4_K_M

# Just prune (no quantization)
./llama-quantize --nash-prune --output-type f16 \
    model-f16.gguf \
    model-pruned-f16.gguf
```

### 7.2 Full Options

```bash
./llama-quantize \
    --nash-prune                    # Enable Nash pruning
    --nash-lambda 0.5               # Redundancy penalty weight
    --nash-threshold 0.1            # Pruning threshold for s_i
    --nash-iterations 100           # Max equilibrium iterations
    --nash-min-heads 2              # Minimum heads per layer
    --nash-max-prune 0.6            # Maximum prune ratio per layer
    --nash-calibration data.txt     # Calibration text file
    --nash-calibration-tokens 1024  # Tokens to use from file
    --nash-verbose                  # Print per-head statistics
    --nash-stats stats.json         # Save detailed stats
    model-f16.gguf \
    model-pruned-Q4_K_M.gguf \
    Q4_K_M
```

### 7.3 Using Pruned Models

No special flags needed! Pruned models work directly:

```bash
# Interactive chat
./llama-cli -m model-pruned-Q4_K_M.gguf -p "Hello!" -cnv

# Server
./llama-server -m model-pruned-Q4_K_M.gguf --port 8080

# Benchmark
./llama-bench -m model-pruned-Q4_K_M.gguf

# Perplexity test
./llama-perplexity -m model-pruned-Q4_K_M.gguf -f wiki.txt
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```cpp
// tests/test-nash-prune.cpp

TEST(NashPrune, EquilibriumConverges) {
    // Create synthetic importance/redundancy matrices
    // Verify solver converges within max iterations
}

TEST(NashPrune, TensorSlicingCorrect) {
    // Create mock attention tensor
    // Apply pruning mask
    // Verify output dimensions and values
}

TEST(NashPrune, GGUFRoundtrip) {
    // Prune model, save to GGUF
    // Load back, verify head counts match
}

TEST(NashPrune, MinHeadsRespected) {
    // Set aggressive threshold
    // Verify min_heads constraint is honored
}
```

### 8.2 Tiny Model Tests

Target models for development (all fit in 4GB RAM):

| Model | Params | Size (F16) | Size (Q4) | Use Case |
|-------|--------|------------|-----------|----------|
| SmolLM-135M | 135M | ~270MB | ~80MB | Fast iteration |
| Qwen2-0.5B | 500M | ~1GB | ~300MB | Realistic test |
| TinyLlama-1.1B | 1.1B | ~2.2GB | ~600MB | Full validation |

### 8.3 Validation Metrics

```bash
# Before pruning
./llama-perplexity -m model-original.gguf -f wiki.txt > ppl_before.txt

# After pruning
./llama-perplexity -m model-pruned.gguf -f wiki.txt > ppl_after.txt

# Compare
# Acceptable: <5% perplexity increase
# Good: <2% perplexity increase
# Excellent: <1% perplexity increase
```

---

## 9. Memory Savings Analysis

### 9.1 Formula

For attention weights (the targets of pruning):

```
Attention memory per layer (FP16):
  wq: n_embd × head_dim × n_head × 2 bytes
  wk: n_embd × head_dim × n_head_kv × 2 bytes  (unchanged in GQA)
  wv: n_embd × head_dim × n_head_kv × 2 bytes  (unchanged in GQA)
  wo: head_dim × n_head × n_embd × 2 bytes

Pruning saves:
  wq: n_embd × head_dim × (n_head - n_head_new) × 2 bytes
  wo: head_dim × (n_head - n_head_new) × n_embd × 2 bytes

With quantization (e.g., Q4_K_M at ~4.5 bpw):
  Multiply by (4.5/16) = 0.28
```

### 9.2 Example Calculations

**Qwen2-0.5B** (n_embd=896, n_head=14, head_dim=64, n_layer=24):

| Component | Original | 50% Prune | Savings |
|-----------|----------|-----------|---------|
| wq per layer | 1.5 MB | 0.75 MB | 0.75 MB |
| wo per layer | 1.5 MB | 0.75 MB | 0.75 MB |
| Total attn | 72 MB | 36 MB | 36 MB |
| With Q4_K_M | 20 MB | 10 MB | 10 MB |

**Llama-3-8B** (n_embd=4096, n_head=32, head_dim=128, n_layer=32):

| Component | Original | 50% Prune | Savings |
|-----------|----------|-----------|---------|
| wq per layer | 32 MB | 16 MB | 16 MB |
| wo per layer | 32 MB | 16 MB | 16 MB |
| Total attn | 2.0 GB | 1.0 GB | 1.0 GB |
| With Q4_K_M | 0.6 GB | 0.3 GB | 0.3 GB |

---

## 10. Implementation Phases

### Phase 1: Foundation ✓
- [x] Create design document
- [ ] Create `src/llama-prune.h` with data structures
- [ ] Create `src/llama-prune.cpp` skeleton
- [ ] Add to build system (CMakeLists.txt)

### Phase 2: Core Algorithm
- [ ] Implement importance computation (imatrix-based)
- [ ] Implement importance computation (calibration-based)
- [ ] Implement redundancy matrix computation
- [ ] Implement Nash equilibrium solver
- [ ] Add unit tests

### Phase 3: Tensor Manipulation
- [ ] Implement attention tensor slicing
- [ ] Handle bias tensors
- [ ] Handle Q-norm tensors
- [ ] Test tensor correctness

### Phase 4: Integration
- [ ] Modify `llama_model_quantize_params`
- [ ] Insert pruning into quantization pipeline
- [ ] Add GGUF metadata output
- [ ] Test GGUF roundtrip

### Phase 5: CLI & Polish
- [ ] Add CLI flags to llama-quantize
- [ ] Add progress reporting
- [ ] Add verbose statistics output
- [ ] Documentation updates

### Phase 6: Validation
- [ ] Test on SmolLM-135M
- [ ] Test on Qwen2-0.5B
- [ ] Test on TinyLlama-1.1B
- [ ] Perplexity benchmarks
- [ ] Memory usage verification

---

## 11. File Reference

### New Files

| File | Purpose |
|------|---------|
| `src/llama-prune.h` | Data structures and API declarations |
| `src/llama-prune.cpp` | Algorithm implementation |
| `tests/test-nash-prune.cpp` | Unit tests |
| `docs/nash-equilibrium-pruning.md` | This document |

### Modified Files

| File | Changes |
|------|---------|
| `include/llama.h` | Add `nash_prune_params` to quantize params |
| `src/llama-quant.cpp` | Insert pruning phase |
| `tools/quantize/quantize.cpp` | Add CLI flags |
| `src/CMakeLists.txt` | Add new source files |
| `src/llama-arch.h` | Add pruning metadata keys |
| `src/llama-arch.cpp` | Register metadata key strings |

### Reference Files (Unchanged)

| File | Relevance |
|------|-----------|
| `src/llama-hparams.h` | Contains n_head_arr (already per-layer) |
| `src/llama-model.cpp` | Model loading (already handles variable heads) |
| `src/llama-graph.cpp` | Attention computation (uses hparams.n_head(il)) |

---

## 12. Troubleshooting

### Q: Pruned model produces garbage output

**Causes:**
1. Tensor slicing offset error
2. Head count metadata mismatch
3. GQA ratio broken (n_head not divisible by n_head_kv)

**Debug:**
```bash
./llama-cli -m model.gguf --verbose-prompt -p "Test" -n 1
# Check for dimension mismatches in output
```

### Q: No memory savings observed

**Causes:**
1. Pruning threshold too high (no heads pruned)
2. Model uses shared heads (MQA/GQA with n_head_kv = n_head)

**Debug:**
```bash
./llama-quantize --nash-prune --nash-verbose model.gguf out.gguf Q4_K_M
# Check pruning statistics output
```

### Q: Nash equilibrium doesn't converge

**Causes:**
1. Learning rate too high (oscillation)
2. Lambda too low (unstable dynamics)
3. Degenerate importance scores

**Solutions:**
- Reduce learning rate: `--nash-lr 0.001`
- Increase lambda: `--nash-lambda 0.8`
- Use more calibration data

### Q: Perplexity significantly degraded

**Causes:**
1. Pruned too aggressively
2. Important heads removed
3. Insufficient calibration data

**Solutions:**
- Raise threshold: `--nash-threshold 0.2`
- Lower max prune ratio: `--nash-max-prune 0.4`
- Use more calibration tokens: `--nash-calibration-tokens 2048`

---

## References

1. Roemmele, M. - Nash equilibrium-based pruning for LLMs (original research)
2. llama.cpp documentation: https://github.com/ggml-org/llama.cpp
3. GGUF format specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

---

*Last updated: 2024-12-30*
*Implementation branch: claude/add-theorizing-llama-cpp-PMdRy*
