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

## Automated Tuning (Python Framework)

The manual benchmark checklist below is the educational reference. The project also includes a **Python framework** that automates the entire tuning pipeline:

```bash
# One command to find the optimal Hermes fallback config
python -m llama_autotune --model ~/models/model.gguf --purpose fallback

# Maximise raw throughput instead
python -m llama_autotune --model ~/models/model.gguf --purpose primary

# Output as JSON for scripting
python -m llama_autotune --model ~/models/model.gguf --json

# Just inspect your hardware
python -m llama_autotune --model ~/models/model.gguf --validate-only
```

### Architecture

```
llama-autotune/
├── llama_autotune/
│   ├── autotune.py        # Main entry point / orchestrator
│   ├── hardware.py        # CPU, RAM, GPU, storage, OS detection
│   ├── build_detector.py  # Recommend best llama.cpp backend (CUDA/HIP/Vulkan/SYCL/CPU)
│   ├── benchmark.py       # Parameter grid runner, OOM detection, TPS parsing
│   ├── scoring.py         # Weighted scoring — throughput vs resource usage
│   ├── recommendations.py # Produce ready-to-use llama-cli commands
│   └── report.py          # Markdown report with tables and command blocks
├── configs/
│   ├── cpu_only.yaml      # Preset for CPU-only hardware
│   ├── nvidia.yaml        # Preset for NVIDIA GPUs
│   ├── amd.yaml           # Preset for AMD GPUs
│   └── intel_gpu.yaml     # Preset for Intel Arc / iGPU
└── reports/               # Generated reports land here
```

### Pipeline (7 phases)

| Phase | Module | What it does |
|-------|--------|--------------|
| **1. Detect** | `hardware.py` | Reads `lscpu`, `nvidia-smi`, `rocminfo`, `vulkaninfo`, `sycl-ls`, `free`, `dmidecode`, `/proc/meminfo`, `lsblk` |
| **2. Recommend build** | `build_detector.py` | Maps detected GPU vendor → optimal CMake backend (`-DGGML_CUDA=ON`, `-DGGML_HIP=ON`, `-DGGML_VULKAN=ON`, `-DGGML_SYCL=ON`) |
| **3. Build grid** | `benchmark.py` | Generates a cross-product of threads × GPU layers × batch × ubatch × context × FA × KV cache × mlock × mmap, pruned to hardware limits |
| **4. Benchmark** | `benchmark.py` | Runs `llama-cli` for each config, parses `llama_print_timings`, detects OOM (`cudaMalloc failed`, segfaults, `std::bad_alloc`) |
| **5. Score** | `scoring.py` | Weighted formula: `prompt_tps × 0.25 + gen_tps × 0.35 + context × 0.15 + resource_penalty × 0.25` (fallback mode penalises CPU hogging) |
| **6. Recommend** | `recommendations.py` | Produces two configs: **primary** (max throughput) and **fallback** (resource-aware for Hermes) |
| **7. Report** | `report.py` | Markdown with hardware table, build recommendation, top configs with copy-paste commands, top-20 results table, OOM/failure log |

### Scoring — What "best" means

The scoring weights differ by purpose:

**Fallback mode (default):**
```
Score = prompt_tps_norm × 0.25
      + gen_tps_norm    × 0.35
      + context_norm    × 0.15
      + resource_score  × 0.25        ← heavy penalty for CPU hogging
      + stability_bonus (32K+ context, mlock)
```

**Primary mode (max throughput):**
```
Score = prompt_tps_norm × 0.35
      + gen_tps_norm    × 0.45
      + context_norm    × 0.15
      + resource_score  × 0.05        ← minimal penalty
```

The resource score penalises thread counts that saturate the CPU. On a 4-core / 8-thread machine, using all 8 threads gets dinged harder in fallback mode because Hermes needs those cores too.

### OOM handling

The benchmark module recognises these failure patterns in stderr:

- `cudaMalloc failed`
- `out of memory`
- `failed to allocate`
- `Segmentation fault` / `SIGSEGV`
- `std::bad_alloc`

When OOMs are detected, the grid is dynamically pruned — higher GPU layers, larger contexts, and bigger batch sizes that caused OOM are removed from the remaining search space.

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

## Benchmark Checklist (Reproducible Commands)

This section is a copy-paste benchmarking workflow for tuning llama.cpp on any hardware. The goal is **not** the highest benchmark numbers — it's finding the best balance between CPU usage, RAM usage, VRAM usage, context length, throughput, and system responsiveness. If you're running llama.cpp as a fallback for Hermes (or another local agent), remember the agent itself needs CPU time and memory too.

### Before You Start

Record your hardware baseline:

```bash
lscpu
free -h
nvidia-smi
./llama-cli --version
```

Note: CPU model, core/thread count, total RAM, GPU model, VRAM, llama.cpp version, CUDA version.

### Monitoring

Keep these running in a second terminal while benchmarking:

```bash
htop                          # CPU usage
watch -n1 free -h             # Memory
watch -n1 nvidia-smi          # GPU
```

### Test Prompt

Use the same prompt for every run to get comparable numbers. Generate enough tokens for stable measurement with `-n 512`:

```text
Write a detailed explanation of how quicksort works, including pseudocode, complexity analysis, and practical optimizations.
```

### Baseline (CPU Only)

```bash
./llama-cli \
  -m MODEL.gguf \
  -ngl 0 \
  -c 32768 \
  -b 256 \
  -ub 128 \
  -t 4 \
  -fa on \
  -n 512 \
  -p "Write a detailed explanation of how quicksort works, including pseudocode, complexity analysis, and practical optimizations."
```

Record: prompt tok/s, generation tok/s, RAM usage.

### GPU Layer Scaling

Test each value. Increase until CUDA OOM, performance plateaus, or VRAM fills completely. Pick the highest value that still leaves VRAM headroom.

```bash
-ngl 0   # CPU only
-ngl 1
-ngl 2
-ngl 3
-ngl 4
-ngl 5
-ngl 6
```

### Context Size Scaling

```bash
-c 4096      # 4K
-c 8192      # 8K
-c 16384     # 16K
-c 32768     # 32K
-c 65536     # 64K
-c 131072    # 128K
```

Record prompt speed, memory usage, and stability. If larger contexts force fewer GPU layers, weigh the trade-off.

### Batch Size Benchmark

```bash
-b 64
-b 128
-b 256
-b 512
-b 1024
```

Larger values improve prompt throughput but increase memory usage. The optimum depends on available RAM and VRAM.

### Micro-Batch Benchmark

```bash
-ub 64
-ub 128
-ub 256
-ub 512
```

Large micro-batches may increase VRAM pressure.

### Thread Count Benchmark

```bash
-t 2
-t 4
-t 6
-t 8
```

Many older CPUs perform nearly as well on physical cores only. Leave spare threads for Hermes, the browser, the OS, and any embedding models.

### CPU Affinity

```bash
taskset -c 0-7   ./llama-cli ...   # All CPUs
taskset -c 0,2,4,6 ./llama-cli ... # Physical cores only
taskset -c 1,3,5,7 ./llama-cli ... # Hyperthreads only
```

llama.cpp also supports built-in affinity (`--cpu-mask`, `--cpu-range`, `--cpu-strict`).

### NUMA (Multi-Socket Systems)

Most laptops have one NUMA node. For multi-socket workstations:

```bash
--numa distribute
--numa isolate
--numa numactl
```

Single-socket laptops typically see no measurable effect.

### mmap Benchmark

```bash
./llama-cli ...        # Default (mmap on)
./llama-cli --no-mmap  # Memory mapping disabled
```

mmap is enabled by default; disabling it can affect load time and paging behavior.

### mlock Benchmark

```bash
./llama-cli ...        # Without
./llama-cli --mlock    # With (locks model into RAM)
```

Useful if enough RAM is available — prevents swapping during long sessions.

### Flash Attention Benchmark

```bash
-fa on
-fa off
```

Flash Attention is generally recommended and required for some KV cache configurations.

### KV Cache Benchmark

```bash
(no flags)             # Default (f16)
-ctk q8_0 -ctv q8_0    # Q8 quantization
-ctk q4_0 -ctv q4_0    # Q4 quantization
```

Quantized KV cache greatly reduces memory usage, enabling larger context windows on low-memory hardware.

### Long Prompt Benchmark

```bash
./llama-cli \
  -m MODEL.gguf \
  -ngl X \
  -c 32768 \
  -b 256 \
  -ub 128 \
  -fa on \
  -t 4 \
  -n 512 \
  -p "Write a detailed explanation of how quicksort works, including pseudocode, complexity analysis, practical optimizations, common mistakes, and real-world implementations."
```

### Code Generation Benchmark

```bash
./llama-cli \
  -m MODEL.gguf \
  -ngl X \
  -c 32768 \
  -b 256 \
  -ub 128 \
  -fa on \
  -t 4 \
  -n 512 \
  -p "Write a complete Python implementation of a threaded web crawler with retry logic, logging, unit tests and documentation."
```

### Conversation Benchmark

```bash
./llama-cli \
  -m MODEL.gguf \
  -ngl X \
  -c 32768 \
  -b 256 \
  -ub 128 \
  -fa on \
  -t 4
```

Chat naturally for several minutes. Watch for memory growth, VRAM creep, CPU saturation, and responsiveness.

### Stress Test

Keep generating until several thousand tokens have been produced. Observe: memory leaks, CUDA OOM, swapping, performance degradation.

### Results Table

| Test | Prompt t/s | Gen t/s | RAM | VRAM | Stable |
|------|-----------:|--------:|----:|-----:|:------:|
| CPU Only | | | | | |
| ngl=2 | | | | | |
| ngl=4 | | | | | |
| ngl=6 | | | | | |
| 16K Context | | | | | |
| 32K Context | | | | | |
| 64K Context | | | | | |
| Batch 128 | | | | | |
| Batch 256 | | | | | |
| Batch 512 | | | | | |
| Threads 4 | | | | | |
| Threads 8 | | | | | |
| mmap Disabled | | | | | |
| mlock Enabled | | | | | |

### Choosing Your Final Configuration

After completing all benchmarks:

1. Choose the highest stable context size.
2. Choose the highest GPU layer count that does **not** exhaust VRAM.
3. Select the smallest thread count that delivers nearly maximum throughput.
4. Choose the largest batch size that does not cause memory pressure.
5. Enable Flash Attention if supported.
6. Use quantized KV cache (`q4_0` or `q8_0`) when long context matters more than the small quality difference.
7. If llama.cpp runs alongside another application (e.g. Hermes AI), prioritize leaving CPU cores and RAM for the primary application rather than maximizing llama.cpp benchmark numbers.

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