# Engineering Schedule: mini-vllm-rs (Nov 19 – Dec 31, 2026)

## Overview & Operating Rules

- **Total Duration:** 6 Weeks (43 Days)
- **Time Commitment:** ~10–12 hours/week (1.5–2 hrs/weekday or focused weekend blocks)
- **Mode:** **Hard Mode** — The owner writes 100% of the code; AI acts exclusively as reviewer, architect, and challenger.
- **Foundational Decision:** Pure `candle-core` tensor operations (no `candle-nn` or pre-built transformers).

---

## Week 1: Raw Tensor Primitives & Forward Pass Baseline (Nov 19 – Nov 25)
*Goal: Load SmolLM-135M weights and execute a single-sequence forward pass using raw tensor math.*

| Date | Focus | Deliverables & Tasks | Verification / Done Criteria |
| :--- | :--- | :--- | :--- |
| **Nov 19 (Thu)** | Workspace Setup & Tensor Basics | Add `candle-core` to `Cargo.toml`. Write basic tensor matrix multiplication, broadcasting, and shape manipulation tests. | `cargo test` passes on basic tensor arithmetic. |
| **Nov 20 (Fri)** | RMSNorm & Stable Softmax | Implement Root Mean Square Normalization ($\text{RMSNorm}$) and numerically stable Softmax ($\text{Softmax}(x - \max(x))$) using pure tensor ops. | Unit tests proving outputs match numerical reference within $10^{-5}$. |
| **Nov 21 (Sat)** | Rotary Position Embeddings (RoPE) | Implement RoPE frequency calculation and complex rotation / interleaved vector math for query ($Q$) and key ($K$). | Unit test verifying position-dependent rotation properties. |
| **Nov 22 (Sun)** | SafeTensors & Weight Loading | Load `SmolLM-135M` `.safetensors` weights via memory mapping (`mmap`). Inspect tensor shapes and data types (`F32` or `BF16`). | Program loads all model weights into memory within < 500ms. |
| **Nov 23 (Mon)** | Self-Attention Block | Implement $Q, K, V$ linear projections, scaled dot-product attention ($\text{Softmax}(QK^T / \sqrt{d_k})V$), and output projection $O$. | Output tensor shape matches `[batch_size, seq_len, hidden_dim]`. |
| **Nov 24 (Tue)** | MLP Block & Full Transformer Loop | Implement SwiGLU / MLP layer ($\text{silu}(W_{gate}x) \cdot W_{up}x \cdot W_{down}$) and loop through all transformer layers to the final LM head. | Full forward pass produces logits of shape `[1, seq_len, vocab_size]`. |
| **Nov 25 (Wed)** | Milestone 1 Verification | Run greedy next-token prediction on a test prompt. | Predicted next token ID matches Hugging Face / Python reference output. |

---

## Week 2: Naive KV-Cache vs. Paged Memory Foundations (Nov 26 – Dec 02)
*Goal: Implement naive KV caching, measure its memory fragmentation, and architect the `BlockManager`.*

| Date | Focus | Deliverables & Tasks | Verification / Done Criteria |
| :--- | :--- | :--- | :--- |
| **Nov 26 (Thu)** | Naive KV Cache | Implement autoregressive token generation storing historical $K$ and $V$ tensors using concatenation (`Tensor::cat`). | Multi-token text generation loop runs and generates 20 coherent tokens. |
| **Nov 27 (Fri)** | Cache Profiling & Bottlenecks | Profile latency per token and memory allocations as generation sequence length grows from 1 to 256. | Document latency curves showing $O(N)$ allocation overhead of `cat`. |
| **Nov 28 (Sat)** | Paged Memory Architecture | Define core types: `PhysicalBlockId(usize)`, `Block` metadata (reference count, slot usage), and block size (16 tokens). | Design reviewed against vLLM SOSP paper (Sections 3 & 4). |
| **Nov 29 (Sun)** | `BlockManager` (Free Pool) | Implement the physical block allocator: pre-allocated memory pool, free-list stack/queue, and block allocation/deallocation methods. | Unit tests verifying zero leaks when allocating and freeing 1,000 blocks. |
| **Nov 30 (Mon)** | `BlockTable` (Page Tables) | Implement per-sequence `BlockTable` mapping logical token indices $0..T-1$ to physical block IDs. | Unit test verifying logical slot to physical block translation math. |
| **Dec 01 (Tue)** | Stress Testing the Allocator | Implement simulated multi-sequence allocation with random prompt and generation lengths. | Allocator handles variable-length requests without external fragmentation. |
| **Dec 02 (Wed)** | Milestone 2 Review | Compare memory footprint of `BlockManager` pool vs. naive static pre-allocation. | Document mathematical proof of $\ge 90\%$ memory efficiency. |

---

## Week 3: Paged Attention Execution Layer (Dec 03 – Dec 09)
*Goal: Execute transformer self-attention reading directly from discrete, non-contiguous KV blocks.*

| Date | Focus | Deliverables & Tasks | Verification / Done Criteria |
| :--- | :--- | :--- | :--- |
| **Dec 03 (Thu)** | Paged Attention Math | Formulate the gather algorithm: given Query $q_t$ and a sequence's `BlockTable`, compute attention over non-contiguous blocks. | Design document outlining memory offsets and tensor indexing. |
| **Dec 04 (Fri)** | Single-Token Paged Attention | Implement the paged attention operator for the decode step (1 query token attending to historical blocks). | Attention output matches naive continuous attention output. |
| **Dec 05 (Sat)** | Prefill Phase Integration | Implement prompt prefill: partition prompt tokens into 16-token physical blocks and populate cache slots. | Prefill accurately populates initial blocks and produces first output token. |
| **Dec 06 (Sun)** | Decode Step Integration | Implement single-token decode: append $k_t, v_t$ to the current block, or allocate a new block when reaching slot 16. | Generates 50 consecutive tokens correctly crossing multiple block boundaries. |
| **Dec 07 (Mon)** | Copy-On-Write (CoW) | Implement block sharing: multiple sequences point to the same physical block with `ref_count > 1`, cloning only when modified. | Unit test verifying shared blocks avoid memory duplication until mutation. |
| **Dec 08 (Tue)** | Bit-Exact Parity Testing | Run property test comparing greedy generation between Naive KV Cache and Paged KV Cache on 5 different prompts. | 100% identical token IDs generated across both engines. |
| **Dec 09 (Wed)** | Milestone 3 Polish | Clean up compiler warnings, check `cargo clippy -D warnings`, and format with `cargo fmt`. | Clean build with zero warnings. |

---

## Week 4: Iteration-Level Continuous Batching (Dec 10 – Dec 16)
*Goal: Build the vLLM-style scheduler supporting dynamic request insertion and completion.*

| Date | Focus | Deliverables & Tasks | Verification / Done Criteria |
| :--- | :--- | :--- | :--- |
| **Dec 10 (Thu)** | Sequence State Machine | Implement `Sequence` (`id`, `tokens`, `status`: `Waiting`, `Running`, `Finished`) and `SequenceGroup`. | State transitions tested with comprehensive unit tests. |
| **Dec 11 (Fri)** | The `Scheduler` Core | Implement `Scheduler::step()`: inspect available blocks, budget tokens, and schedule active sequences. | Scheduler accurately prioritizes running vs waiting requests based on memory. |
| **Dec 12 (Sat)** | Dynamic Batch Execution | Interleave new request prefills with active request decode steps in a unified execution batch. | Engine generates tokens for 3 concurrent requests of different lengths. |
| **Dec 13 (Sun)** | Instant Block Reclamation | When a sequence emits `<|endoftext|>` or reaches max tokens, immediately return its blocks to the free pool during the step. | Memory pool recovers 100% of freed blocks instantly without engine pauses. |
| **Dec 14 (Mon)** | Preemption & Swapping | Handle memory exhaustion: if free blocks hit 0, suspend the lowest priority request back to `Waiting`. | Stress test: engine handles memory over-subscription gracefully without crashing. |
| **Dec 15 (Tue)** | Multi-Request Concurrency Test | Run 10 synthetic requests with randomized arrival times and generation lengths. | All 10 requests complete with correct outputs and zero deadlocks. |
| **Dec 16 (Wed)** | Milestone 4 Review & CI | Set up GitHub Actions CI workflow to run formatting, clippy, and unit tests on every push. | GitHub Actions CI badge turns green. |

---

## Week 5: End-to-End Pipeline & Tokenizer Integration (Dec 17 – Dec 23)
*Goal: Wire Byteforge, async request queues, and streaming text output into a functional server.*

| Date | Focus | Deliverables & Tasks | Verification / Done Criteria |
| :--- | :--- | :--- | :--- |
| **Dec 17 (Thu)** | Byteforge Integration | Link `byteforge` into `mini-vllm-rs` via path/git dependency. Use Byteforge for prompt tokenization and output detokenization. | Engine ingests raw `&str` and outputs raw `String`. |
| **Dec 18 (Fri)** | Async Request Ingress | Build an actor-style architecture: async Tokio tasks receive requests and push to an MPSC channel; the engine thread polls and batches. | Non-blocking communication between I/O and compute threads. |
| **Dec 19 (Sat)** | Streaming Token Channels | Implement per-request streaming output channels (`tokio::sync::mpsc`). Stream tokens back to caller in real-time as decoded. | CLI prints tokens streaming in real-time as they are generated. |
| **Dec 20 (Sun)** | Stop Criteria & Special Tokens | Add EOS token handling, user-defined stop sequences, and `max_new_tokens` enforcement. | Generation terminates reliably across edge-case prompt completions. |
| **Dec 21 (Mon)** | Concurrent Load Testing | Test 20 concurrent streaming requests simultaneously. Measure throughput (tokens/sec) and time-to-first-token (TTFT). | Engine serves 20 concurrent streams without memory leaks. |
| **Dec 22 (Tue)** | Memory Profiling & Leak Check | Profile process RSS memory under sustained load using system tools (e.g. `htop`, Heaptrack). | Memory remains stable and does not grow unboundedly over 1,000 requests. |
| **Dec 23 (Wed)** | Milestone 5 Polish | Refactor error types using `thiserror` / `anyhow`. Ensure idiomatic Rust error propagation. | Zero `unwrap()` calls in production engine code paths. |

---

## Week 6: Prefix Caching, Benchmarking & Portfolio Polish (Dec 24 – Dec 31)
*Goal: Implement Radix-Tree prefix caching, produce benchmark graphs, and finalize the portfolio release.*

| Date | Focus | Deliverables & Tasks | Verification / Done Criteria |
| :--- | :--- | :--- | :--- |
| **Dec 24 (Thu)** | Radix-Tree Prefix Cache | Implement a Radix-Tree in Rust to index token prefixes and map common prompt sequences to cached `PhysicalBlockId`s. | Radix-Tree unit tests pass (insert, lookup, longest prefix match). |
| **Dec 25 (Fri)** | Prefix Caching in Serving Loop | When new requests share a common system prompt prefix, skip prefill compute and reuse existing physical blocks directly. | Time-to-First-Token (TTFT) for repeated prompts drops by $\ge 70\%$. |
| **Dec 26 (Sat)** | Criterion Throughput Benchmarks | Write Criterion benchmarks measuring decode throughput (tokens/sec) across batch sizes 1, 4, 8, 16. | Reproducible Criterion benchmark reports in `target/criterion`. |
| **Dec 27 (Sun)** | Fragmentation Benchmark | Create a benchmark script comparing memory consumption: Naive contiguous allocation vs. Paged KV under concurrent load. | Data generated demonstrating 60–80% memory savings. |
| **Dec 28 (Mon)** | Architecture Documentation | Write `docs/ARCHITECTURE.md` with detailed diagrams of the `BlockManager`, `Scheduler`, and Paged Attention execution flow. | Comprehensive, publication-grade documentation. |
| **Dec 29 (Tue)** | README & Portfolio Showcase | Update root `README.md` with benchmark plots, architecture diagrams, and instructions on how to run the engine. | README renders cleanly with clear instructions and visuals. |
| **Dec 30 (Wed)** | Final Code Audit & Clippy Clean | Final pass: `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`, full test suite pass. | Cleanest possible code quality score. |
| **Dec 31 (Thu)** | Milestone 6 Release (v0.1.0) | Tag Git release `v0.1.0`. Celebrate having built an operating-system-grade LLM serving engine in Rust from scratch! | Working serving engine ready for Big Tech resumes and vLLM PRs. |
