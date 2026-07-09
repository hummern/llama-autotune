# llama-autotune

Automated hyperparameter tuning for [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF models.

Find the optimal inference parameters — context size, batch size, thread count, GPU offload layers, and KV cache quantization — for any GGUF model on your hardware.

## Features

- **Benchmark-driven tuning** — measures tokens/sec across a configurable parameter grid
- **Per-model profiles** — persists optimal settings per model file, so you tune once
- **Hardware-aware** — adapts to your CPU cores, RAM, and GPU memory automatically
- **llama.cpp native** — calls the `llama-cli`/`llama-bench` binaries directly, no wrappers

## Installation

```bash
git clone https://github.com/hummern/llama-autotune.git
cd llama-autotune
pip install -r requirements.txt
```

**Prerequisites:** A working [llama.cpp](https://github.com/ggml-org/llama.cpp) build with `llama-cli` and `llama-bench` on your `PATH`.

## Quick Start

```bash
# Tune a single model
python -m llama_autotune ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Tune all GGUF files in a directory
python -m llama_autotune --scan ~/models/

# Show the best settings for a previously tuned model
python -m llama_autotune --show ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## How It Works

1. **Discover hardware** — detects CPU cores, total RAM, and GPU presence/memory
2. **Build a parameter grid** — generates sensible ranges for context size, batch size, threads, GPU layers, and KV cache type
3. **Benchmark each combination** — runs short `llama-bench` sessions and records tokens/sec
4. **Score and rank** — weights throughput, memory safety, and prompt processing speed
5. **Save the profile** — stores the winner alongside the model file

## Configuration

Tuning behavior is controlled via a `config.yaml` or CLI flags:

| Parameter | Flag | Description |
|-----------|------|-------------|
| `max_context` | `--max-ctx` | Upper bound for context length search (default: hardware-limit) |
| `min_context` | `--min-ctx` | Lower bound (default: 512) |
| `thread_step` | `--thread-step` | Thread count increment (default: 2) |
| `prompt_tokens` | `--prompt-len` | Token count for benchmark prompts (default: 512) |
| `gen_tokens` | `--gen-len` | Token count for generation benchmark (default: 128) |
| `timeout` | `--timeout` | Max seconds per benchmark run (default: 30) |

## Output

```
$ python -m llama_autotune ~/models/Llama-3.1-8B-Q4_K_M.gguf

  llama-autotune v0.1.0
  Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
  Hardware: 16 cores, 32 GB RAM, RTX 4070 (12 GB VRAM)

  Searching 48 combinations...
  ===========> 100%

  Best result:
    ctx: 8192   batch: 512   threads: 12   gpu-layers: 33   kv-cache: q8_0
    prompt: 1240.5 tok/s   generation: 58.3 tok/s

  Profile saved to ~/models/Llama-3.1-8B-Q4_K_M.gguf.autotune.yaml
```

## License

MIT