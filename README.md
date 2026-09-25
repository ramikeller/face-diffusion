# face-diffusion

A from-scratch DDPM (denoising diffusion probabilistic model) in Rust, trained on celebrity faces. No pretrained weights, no diffusion library — the network learns to reverse a fixed noising process, then sampling starts from pure random noise and iteratively denoises it into a novel 64×64 face.

Deliberately small (64×64 images, a 2-level U-Net, ~1.1M parameters) so it trains fast enough on a single GPU to actually iterate on, rather than aiming for production-quality output. Example output after 23,300 training steps (~14 hours on an M4):

![Example output grid](docs/example_output.png)

Still soft/painterly rather than sharp — that's the tiny model's ceiling, not a bug. See [Results](#results) below.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1 | [Cargo.toml](Cargo.toml) | [Candle](https://github.com/huggingface/candle) (pure-Rust ML framework, Metal GPU backend) |
| 2 | [src/data.rs](src/data.rs) | Loads face images, center-crops + resizes to 64×64, normalizes to `[-1, 1]` |
| 3 | [src/schedule.rs](src/schedule.rs) | Linear β noise schedule; `q_sample` (forward diffusion) and `p_sample` (one reverse step) |
| 4 | [src/unet.rs](src/unet.rs) | 2-level U-Net noise predictor with sinusoidal timestep conditioning |
| 5 | [src/train.rs](src/train.rs) | Training loop: predict the noise added to a randomly-noised image, MSE loss, AdamW |
| 6 | [src/sample.rs](src/sample.rs) | Reverse sampling loop (pure noise → image) and grid-PNG export |
| — | [src/main.rs](src/main.rs) | CLI entry point wiring it all together |

## Setup

Requires the Rust toolchain (`rustc`/`cargo`) and, for the dataset step, a throwaway Python virtualenv.

### 1. Get the dataset

Not part of the Rust pipeline — a one-off export of 20,000 images from a landmark-aligned CelebA mirror into `data/faces/`:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install huggingface_hub
python3 scripts/export_faces.py
```

This must be the *aligned* CelebA (eyes/nose/mouth at consistent pixel positions across the dataset) — an unaligned mirror will train just fine numerically but the output will never form a coherent face, since a small unconditional model has no other signal telling it where facial features should be. See `scripts/export_faces.py` for the exact source used.

### 2. Build

```sh
cargo build --release
```

candle-core/candle-nn are pinned to a specific upstream commit and patched locally in `vendor/candle/` — see [Known issues](#known-issues-fixed-locally) for why.

## Usage

```sh
# Train. Resumes automatically from checkpoints/unet.safetensors if it exists.
# Saves a checkpoint every 500 steps (configurable in main.rs) and at the end.
cargo run --release -- <steps>       # e.g. cargo run --release -- 5000

# Generate a grid of images from the current checkpoint.
cargo run --release -- sample <n>    # e.g. cargo run --release -- sample 16
# writes samples/grid.png
```

At ~0.4 steps/s on an M4 (batch size 64), expect roughly 1,400-1,500 training steps per hour.

## Results

Loss plateaus very early (within a few hundred steps) to a floor around 0.02-0.05 and barely moves after that — this is an inherent property of the MSE noise-prediction objective, not a sign that training has stalled. **Loss is a poor signal for visual progress here; you have to actually sample and look.** Empirically, with this exact setup:

- **~1,500 steps**: pure color/skin-tone blobs, no structure at all.
- **~7,500 steps**: real (if blurry) facial structure starts appearing — hair, eye regions, rough face outlines.
- **~23,000 steps**: consistently recognizable, if soft/impressionistic, faces.

Further training keeps helping but with diminishing returns; the tiny architecture (32 base channels, 2 levels) caps how sharp it can ever get regardless of training time. The natural next lever is model capacity, not more steps — e.g. doubling to 64 base channels (~4x compute).

## Known issues (fixed locally)

Three real bugs in candle's Metal backend surfaced while building this, none yet fixed upstream at the commit this project pins to:

1. **Build failure on Apple Silicon stable Rust**: an unstable NEON fp16 intrinsic in candle-core's CPU backend. Fixed upstream since (`vendor/candle` is pinned to a commit that already includes that fix).
2. **Silently wrong gradients**: candle-core's Metal conv2d backward pass fed a non-contiguous tensor into im2col/gemm when computing the weight gradient, producing an incorrect gradient with no error — training would have silently diverged. Patched locally in `vendor/candle/candle-core/src/backprop.rs` (`.contiguous()` before the weight-gradient conv calls). See [huggingface/candle#3839](https://github.com/huggingface/candle/pull/3839) (unmerged at time of writing).
3. **Unbounded Metal buffer-pool memory growth**: long sequences of GPU ops (e.g. the 400-step sampling loop) could grow wired memory without bound and eventually crash with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Also patched from the same upstream PR, in `vendor/candle/candle-core/src/metal_backend/` and `vendor/candle/candle-metal-kernels/src/metal/commands.rs`.

A fourth bug was in this project's own code, not candle's: `src/sample.rs`'s reverse loop feeds each step's output into the next step's forward pass through the U-Net's trainable weights, which still builds an autodiff graph even though sampling never calls `.backward()`. Left attached, that graph - and every prior step's retained activations - grew across all 400 sequential steps until memory was exhausted. Fixed with `.detach()` each step.

`vendor/candle/` is a trimmed copy of just the crates this project depends on (candle-core, candle-nn, candle-metal-kernels), not the full upstream monorepo.
