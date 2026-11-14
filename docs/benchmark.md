# Running the benchmark

The benchmark steps many worlds in parallel with random actions and prints
world-steps per second. Run every command from the repository root.

## 1. Get the code and install

```bash
git fetch origin
git checkout initial-world
git pull
uv sync
```

`uv sync` creates `.venv/` inside the repository and installs everything there.

## 2. Check that the GPU is seen

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

This should print `True` and the name of your GPU. If it prints `False`, the
benchmark still runs, but on the CPU.

## 3. Run the tests

```bash
uv run pytest
```

Expect `13 passed` on a machine with a GPU (`12 passed, 1 skipped` without one).

## 4. Run the benchmark

Default settings (256 worlds, 1 agent, 64x48 view, GPU if available):

```bash
uv run python scripts/benchmark.py
```

More worlds in parallel:

```bash
uv run python scripts/benchmark.py --num-envs 1024
uv run python scripts/benchmark.py --num-envs 4096
```

Same run on the CPU, for comparison:

```bash
uv run python scripts/benchmark.py --num-envs 1024 --device cpu
```

Each run prints one line, for example:

```
637 world-steps/s on cpu (64 worlds, 1 agent(s), 64x48 view)
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-envs` | 256 | Worlds simulated in parallel |
| `--steps` | 50 | Timed steps (one extra warm-up step is not timed) |
| `--n-agents` | 1 | Agents per world; each one renders its own view |
| `--width` | 64 | View width in pixels |
| `--height` | 48 | View height in pixels |
| `--device` | GPU if available | `cpu`, `cuda`, `cuda:1`, ... |

## If the GPU runs out of memory

Memory grows with `num-envs x n-agents x width x height`. Lower `--num-envs`
first:

```bash
uv run python scripts/benchmark.py --num-envs 512
```
