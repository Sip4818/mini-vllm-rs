# mini-vllm-rs

[![Rust](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Hard_Mode_Active-red.svg)](agent.md)

**mini-vllm-rs** is a high-throughput, low-latency Large Language Model (LLM) serving engine engineered from first principles in idiomatic Rust.

Inspired by UC Berkeley's [vLLM](https://github.com/vllm-project/vllm) architecture (*"Efficient Memory Management for Large Language Model Serving with PagedAttention"*, Kwon et al., 2023), this project implements an operating-system-style virtual memory manager for Key-Value (KV) caches and an iteration-level continuous batching scheduler.

---

## The Problem: Memory Fragmentation in LLM Serving

In standard autoregressive inference, the KV cache grows dynamically with each generated token. Naive serving engines pre-allocate contiguous memory buffers for each request based on the model's `max_context_length` (e.g. 4,096 tokens). 

This introduces critical system bottlenecks:
1. **Internal Fragmentation:** A user generating a 50-token response locks 4,046 empty token slots in memory that no other request can access (wasting up to 60–80% of RAM/VRAM).
2. **External Fragmentation:** Dynamic allocations of variable sizes fragment memory addresses, triggering Out-Of-Memory (OOM) errors even when ample total memory remains.
3. **Head-of-Line Request Blocking:** Static batching forces fast requests to wait idle until the slowest request in the batch completes.

---

## Core Architecture

`mini-vllm-rs` eliminates memory fragmentation and maximizes throughput via three primary subsystems:

```
[ Incoming Client Requests ]
             │
             ▼
┌─────────────────────────────┐
│ Continuous Batching         │  ◄── Iteration-level scheduling: Promotes Waiting
│ Scheduler                   │      sequences, dispatches decode steps, and
└─────────────┬───────────────┘      handles preemption without head-of-line blocking.
              │
              ▼
┌─────────────────────────────┐
│ BlockManager                │  ◄── Virtual Memory Page Table: Maps logical token
│ (Paged KV Memory Pool)      │      positions to non-contiguous physical blocks (16 tokens/block).
└─────────────┬───────────────┘      Supports Copy-On-Write (CoW) and instant recycling.
              │
              ▼
┌─────────────────────────────┐
│ Paged Transformer Execution │  ◄── Executes prefill and decode passes reading from
│ (`candle` backend)          │      discrete memory blocks across CPU, CUDA, and Metal.
└─────────────────────────────┘
```

### 1. PagedAttention & The `BlockManager`
- Treats memory as a pool of discrete, fixed-size physical pages (16 tokens per block).
- Maintains a per-sequence **Page Table** that translates contiguous sequence positions into scattered physical block IDs.
- **Copy-On-Write (CoW):** Identical prompt prefixes share physical memory blocks until divergent tokens are generated.

### 2. Continuous / Iterative Batching
- Replaces coarse request-level batching with fine-grained iteration-level dispatching.
- Incoming prompts (prefill phase) join the engine immediately at the next decode step.
- Finished sequences release physical blocks back to the free pool without waiting for the batch to finish.

### 3. Hardware-Agnostic Execution Layer
- Built on top of Hugging Face's minimalist `candle` tensor framework.
- Decouples systems scheduling from underlying hardware: runs deterministically on **CPU (AVX/NEON)**, with native support for **NVIDIA GPUs (CUDA)** and **Apple Silicon (Metal)**.

---

## Engineering Methodology (Hard Mode)

As outlined in [**`agent.md`**](agent.md), this codebase is built **100% by hand** from first principles. AI assistants act exclusively as challenging systems architects—stress-testing design decisions, memory layouts, and algorithmic complexities—without generating implementation code.

---

## Roadmap (Milestones to Dec 31, 2026)

- [ ] **Milestone 1 — Baseline Autoregressive Loop:** Model loading (`.safetensors`), greedy decode, and unpaged reference KV cache.
- [ ] **Milestone 2 — BlockManager & Virtual Memory:** Block allocation pools, logical-to-physical block tables, and ref-count tracking.
- [ ] **Milestone 3 — Paged Attention Forward Pass:** Non-contiguous KV cache attention operator; bit-exact equivalence testing.
- [ ] **Milestone 4 — Continuous Batching Scheduler:** Request state machine (`Waiting`, `Running`, `Finished`) and dynamic step dispatching.
- [ ] **Milestone 5 — End-to-End Async Pipeline:** Async request ingress, token streaming, and zero-leak memory lifecycle.
- [ ] **Milestone 6 — Prefix Caching & Benchmarking:** Radix-tree prompt sharing, throughput benchmarks (tokens/sec), and memory footprint profiling.

---

## Development

Check formatting and linter:
```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Run test suite:
```bash
cargo test --all-features
```

---

## References

1. Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *"Efficient Memory Management for Large Language Model Serving with PagedAttention."* ACM SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
2. [vLLM Project](https://github.com/vllm-project/vllm)
3. [Candle: Minimalist ML framework for Rust](https://github.com/huggingface/candle)

---

## License

Dual-licensed under [MIT](LICENSE) or Apache-2.0 at your option.
