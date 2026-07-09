# llama-autotune

Automated benchmarking and parameter tuning for [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF models — designed specifically for configuring llama.cpp as a **local fallback inference engine** for [Hermes AI](https://hermes-agent.nousresearch.com/).

## Why This Exists

When the cloud API is down, Hermes needs a local model that works. But the goal is **not** maximum llama.cpp throughput. The goal is:

- Keep Hermes responsive — reserve CPU and RAM for the host agent
- Still have a usable local LLM when the API goes offline
- Support a **32K context window** on modest hardware
- Make the most of limited VRAM to reduce system RAM pressure

Normal Hermes workflow:

```
Hermes
    │
    ├── API available
    │      └── Use remote model (fast, no local resources)
    │
    └── API unavailable
           └── Fall back to llama.cpp locally (tuned for minimal impact)
```

Since llama.cpp spends most of its time idle as a fallback, every parameter choice should err on the side of leaving resources for Hermes, not chasing benchmark numbers.

## Reference Hardware & Results

This tool was developed and validated on an older laptop. The benchmark data below serves as a representative case study — the same methodology applies to any hardware.

| Component | Detail |
|-----------|--------|
| **Laptop** | Multicom Xishan W230S (W230SS-CFB5) |
| **CPU** | Intel Core i7-4710MQ — 4 cores / 8 threads, 2.5 GHz base, 3.5 GHz turbo |
| **RAM** | 10 GB DDR3 |
| **GPU** | NVIDIA GeForce GTX 860M — 2 GB VRAM, CUDA CC 5.0 |
| **OS** | Ubuntu 24.04 |
| **llama.cpp** | Official CUDA build |
| **Model** | Ornith-1.0-9B-MTP-Q4_K_M.gguf ([HuggingFace](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF)) |

### Final Recommended Command

```bash
./llama-cli \
  -m /path/to/Ornith-1.0-9B-MTP-Q4_K_M.gguf \
  -ngl 4 \
  -c 32768 \
  -b 256 \
  -ub 128 \
  -ctk q4_0 \
  -ctv q4_0 \
  -fa on \
  -t 4 \
  --mlock
```

**Performance on this hardware:**

| Metric | Value |
|--------|-------|
| Prompt evaluation | ≈ 8.5–9 tok/s |
| Generation | ≈ 2.5 tok/s |
| Context window | 32,768 tokens |

## Parameter-by-Parameter Rationale

### `-ngl 4` — GPU Layers

Only 2 GB VRAM available. Tested 0 through 6 layers:

| Layers | Result |
|---------|--------|
| 0 | CPU only — highest RAM usage |
| 2 | Slight improvement |
| **4** | **Best balance — frees RAM, leaves VRAM for CUDA buffers** |
| 6 | Marginally faster, no practical benefit |
| >6 | Out of VRAM |

4 layers performs nearly the same as 6 while preserving GPU memory headroom.

### `-c 32768` — Context Size

| Context | Result |
|---------|--------|
| 16K | ✓ Easy |
| **32K** | **✓ Stable — sweet spot** |
| 64K | Works but forces fewer GPU layers |
| 128K | Out of memory |

32K is the realistic upper bound for this hardware class.

### `-b 256` / `-ub 128` — Batch Sizes

Tested 64–1024. 512 was only slightly faster. Larger batches increased memory usage. 256 offers near-identical throughput with reduced memory pressure.

### `-ctk q4_0` / `-ctv q4_0` — KV Cache Quantization

Essential for long contexts on low-memory machines. Reduces RAM usage enough to make 32K practical on 10 GB, with a small quality trade-off. Flash Attention (`-fa on`) is required when using quantized KV cache in current llama.cpp builds.

### `-t 4` — Threads

CPU: 4 physical cores, 8 threads.

| Threads | Prompt (tok/s) | Generation (tok/s) |
|---------|----------------|---------------------|
| 4 | ≈ 8.8 | ≈ 2.6 |
| 6 | ≈ 8.9 | ≈ 2.5 |
| 8 | ≈ 9.1 | ≈ 2.5 |

Using all 8 threads improves prompt throughput by only a few percent while consuming the entire CPU. **4 threads leaves half the CPU for Hermes.**

### `--mlock`

Locks the model into RAM to prevent swapping during long sessions. Omit if memory-constrained.

## Full Benchmark Matrix

Every combination tested:

| Parameter | Values Tested |
|-----------|---------------|
| GPU layers | 0, 1, 2, 3, 4, 5, 6 |
| Context sizes | 512, 1K, 2K, 4K, 32K, 64K, 128K |
| Batch sizes | 64, 128, 256, 512, 1024 |
| Thread counts | 4, 6, 8 (+ CPU affinity, physical-only, HT-only, OpenMP affinity) |
| Memory modes | default mmap, `--no-mmap`, `--mlock` |

## Lessons Learned

For constrained hardware running alongside Hermes:

1. **Don't chase maximum benchmark numbers.** Stability and headroom matter more.
2. **Leave CPU resources for the host.** Hermes needs threads too.
3. **Even 2 GB of GPU offload is worthwhile.** Every layer offloaded frees system RAM.
4. **Quantized KV cache is non-negotiable** for long contexts on low memory.
5. **32K context is a realistic target** on older hardware.
6. **0.2 tok/s is not worth a full CPU.** A responsive Hermes beats a maxed-out llama.cpp.

## Recommended Use Cases

- Hermes AI fallback inference
- Offline coding assistance
- Emergency local inference
- Travel laptops with limited connectivity
- Older gaming laptops repurposed as AI workstations
- Small home servers

This is **not** intended to replace a cloud API. It provides a dependable local fallback that keeps Hermes functional when remote inference is unavailable.

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Hermes AI](https://hermes-agent.nousresearch.com/)
- [Ornith GGUF on HuggingFace](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF)

## License

MIT