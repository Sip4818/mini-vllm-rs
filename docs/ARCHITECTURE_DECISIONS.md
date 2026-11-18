# Architecture Decision Records (ADRs)

## ADR 001: Core Tensor Engine Decision (Pure `candle-core`)

- **Status:** Accepted
- **Date:** 2026-11-17
- **Author:** Repository Owner

### Context
`candle` offers both low-level tensor primitives (`candle-core`) and higher-level abstractions (`candle-nn` for layers, `candle-transformers` for pre-built model architectures like LLaMA / SmolLM).

### Decision
The engine will **strictly use `candle-core` raw tensor operations**. We will **NOT** use `candle-nn` layers or pre-built model architectures from `candle-transformers`.

### Rationale & Trade-offs
1. **First-Principles Mastery:** All mathematical layers (RMSNorm, Rotary Position Embeddings / RoPE, and Self-Attention projections) will be implemented directly using raw tensor math, matrix operations, and broadcasting.
2. **Custom Memory Layouts:** Building layers manually allows direct control over how weights and intermediate activations interact with non-contiguous Paged KV blocks without being constrained by third-party abstractions.
3. **Engineering Cost:** Requires implementing numerical stability safeguards manually (e.g., subtracting max values in Softmax, epsilon in RMSNorm) and managing tensor shapes and strides explicitly.
