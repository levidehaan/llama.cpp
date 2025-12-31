# Nash Equilibrium-Based Attention Head Pruning

This document describes the experimental Nash equilibrium pruning feature in llama.cpp, which uses game-theoretic principles to intelligently prune attention heads from transformer models.

## Overview

Nash pruning models each attention head as a strategic player in a non-cooperative game. Each head's "utility" depends on its individual importance to the model minus a penalty for being redundant with other heads. The algorithm finds the Nash equilibrium of this game, where heads with low participation values are candidates for removal.

This approach automatically balances:
- **Head importance**: How much each head contributes to model quality
- **Head redundancy**: How similar/replaceable a head is relative to others

## Algorithm

### Game Setup

For each layer, we define a game where:
- **Players**: Each attention head `i` in the layer
- **Strategy**: Participation level `s_i ∈ [0, 1]` (1 = fully participate, 0 = pruned)
- **Utility function**: `U_i(s) = c_i * s_i - λ * s_i * Σ_j(s_j * r_ij)`

Where:
- `c_i`: Head importance score (from imatrix calibration or heuristics)
- `r_ij`: Redundancy between heads i and j
- `λ`: Regularization parameter (default: 0.3)

### Nash Equilibrium Solution

The equilibrium is found via projected gradient ascent:

```
gradient = c_i - λ * Σ_j(s_j * r_ij)
s_i ← clamp(s_i + lr * gradient, 0, 1)
```

Heads with equilibrium `s_i < threshold` (default: 0.4) are marked for pruning.

### Importance Computation

Head importance can be computed from:

1. **imatrix calibration data** (recommended): Uses attention output weight (wo) importance values. The imatrix stores per-row importance, which maps directly to head dimensions since wo has shape `[head_dim * n_heads, n_embd]`.

2. **Heuristic fallback**: U-shaped layer importance (early/late layers more important) with per-head variation.

## Usage

### Step 1: Generate Calibration Data

```bash
# Create a calibration text file with representative text
cat > calibration.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
Machine learning models require careful calibration.
Programming languages like Python and C++ are used to build software.
EOF

# Generate imatrix
./build/bin/llama-imatrix \
    -m model.gguf \
    -f calibration.txt \
    -o imatrix.dat
```

### Step 2: Apply Nash Pruning During Quantization

```bash
./build/bin/llama-quantize \
    --nash-equilibrium \
    --nash-prune-ratio 0.15 \
    --imatrix imatrix.dat \
    --allow-requantize \
    model.gguf \
    model-nash-pruned.gguf \
    Q8_0
```

Parameters:
- `--nash-equilibrium`: Enable Nash-based head pruning
- `--nash-prune-ratio`: Target fraction of heads to prune (0.0-1.0)
- `--imatrix`: Optional calibration data for accurate importance
- `--allow-requantize`: Required when input is already quantized

## Performance Results

Benchmarks on Qwen2-0.5B with 6.2% head pruning (21 of 336 heads from layers 5, 16, 22):

### Speed Improvement

| Metric | Original | Nash-Pruned | Change |
|--------|----------|-------------|--------|
| Prompt Processing (pp128) | 679.48 t/s | 771.35 t/s | **+13.5%** |
| Token Generation (tg64) | 63.39 t/s | 64.58 t/s | **+1.9%** |

### Model Size

| Metric | Original | Nash-Pruned | Reduction |
|--------|----------|-------------|-----------|
| File Size | 507 MiB | 504 MiB | **3 MiB (0.6%)** |
| Parameters | 494.03M | 491.28M | **2.75M (0.6%)** |

### Memory Usage

| Memory Type | Original | Nash-Pruned | Reduction |
|-------------|----------|-------------|-----------|
| Total Host Memory | 1183 MiB | 1156 MiB | **27 MiB (2.3%)** |
| Model Memory | 500 MiB | 497 MiB | **3 MiB** |
| Context Memory | 384 MiB | 360 MiB | **24 MiB** |

## Quality Considerations

### What Works Well

- Models remain coherent and functional after pruning
- Middle layers (5-20 in a 24-layer model) are most prunable
- GQA ratios are preserved (Q heads and KV heads pruned proportionally)

### Known Limitations

1. **Quality degradation**: Aggressive pruning (>15%) may impact output quality
2. **Task-specific**: Heads important for specific capabilities may be pruned if not represented in calibration data
3. **No perplexity testing**: Current implementation doesn't automatically evaluate quality loss

### Recommended Calibration Data

For best results, calibration text should include:
- Diverse writing styles (formal, casual, technical)
- Various topics the model will be used for
- Both short and long passages
- Code snippets (if model will be used for coding)

## Technical Details

### Files Modified

- `src/llama-prune.h`: Public API and types
- `src/llama-prune.cpp`: Core algorithm implementation
- `src/llama-quant.cpp`: Integration with quantization pipeline
- `src/llama-model.cpp`: Per-layer head count support
- `src/models/qwen2.cpp`: Per-layer KV head support in graph building

### GGUF Metadata

Pruned models store per-layer head counts in GGUF metadata:
- `<arch>.attention.head_count_layers`: Array of Q head counts per layer
- `<arch>.attention.head_count_kv_layers`: Array of KV head counts per layer

### Supported Architectures

Currently tested with:
- Qwen2

Should work with other architectures that use standard attention patterns, but may require updates to graph building code for per-layer head support.

## Experimental Status

This feature is experimental. Key areas for future improvement:

1. Automatic perplexity evaluation before/after pruning
2. Better redundancy estimation using actual activation patterns
3. Support for more model architectures
4. Iterative pruning with quality checkpoints

## References

- Nash Equilibrium: John Nash's concept of strategic equilibrium in game theory
- Attention Head Pruning: Related work on importance-based head pruning
- imatrix: llama.cpp's importance matrix for quantization calibration
