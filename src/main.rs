mod data;
mod ema;
mod sample;
mod schedule;
mod train;
mod unet;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use data::FaceDataset;
use ema::Ema;
use schedule::NoiseSchedule;
use std::path::Path;
use train::TrainConfig;
use unet::UNet;

const IMAGE_SIZE: usize = 64;
const TIMESTEPS: usize = 400;
const CHECKPOINT_PATH: &str = "checkpoints/unet.safetensors";
const EMA_CHECKPOINT_PATH: &str = "checkpoints/unet_ema.safetensors";
/// Per-step EMA decay. Effective averaging window is ~1/(1-decay) = 1000
/// steps (half-life ~700): long enough to smooth out minibatch jitter,
/// short enough that the EMA tracks progress within a few-thousand-step run.
const EMA_DECAY: f64 = 0.999;

/// Tries Metal (Apple GPU), then CUDA (NVIDIA GPU), then falls back to CPU.
/// `new_metal`/`new_cuda` compile on every platform but return an `Err` at
/// runtime if that backend wasn't enabled via Cargo features (see
/// Cargo.toml), so this works unmodified regardless of which platform or
/// feature set the project was built with.
fn select_device() -> Device {
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }
    Device::Cpu
}

fn main() -> anyhow::Result<()> {
    let device = select_device();
    println!("Using device: {device:?}");
    if matches!(device, Device::Cpu) {
        println!(
            "No GPU backend available (or not compiled in) - running on CPU, \
             which will be noticeably slower than Metal/CUDA for this model."
        );
    }

    let schedule = NoiseSchedule::new(TIMESTEPS);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let unet = UNet::new(vb)?;

    // A second, identically-shaped U-Net holding the EMA of the weights
    // above. Never trained directly; `Ema::update` blends it toward `unet`.
    let mut ema_varmap = VarMap::new();
    let ema_vb = VarBuilder::from_varmap(&ema_varmap, DType::F32, &device);
    let ema_unet = UNet::new(ema_vb)?;

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("sample") => {
            let batch = args
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);
            let raw = args.next().as_deref() == Some("raw");

            // Prefer the EMA weights; `sample <n> raw` uses the live
            // training weights instead, for comparison.
            let model = if !raw && Path::new(EMA_CHECKPOINT_PATH).exists() {
                ema_varmap.load(EMA_CHECKPOINT_PATH)?;
                println!("Loaded EMA checkpoint from {EMA_CHECKPOINT_PATH}");
                &ema_unet
            } else {
                varmap.load(CHECKPOINT_PATH)?;
                println!("Loaded checkpoint from {CHECKPOINT_PATH}");
                &unet
            };

            let images = sample::sample(model, &schedule, batch, IMAGE_SIZE, &device)?;

            let out_path = Path::new("samples/grid.png");
            sample::save_grid(&images, out_path, 4)?;
            println!("Saved {batch} samples to {}", out_path.display());
        }
        other => {
            let steps = other.and_then(|s| s.parse::<usize>().ok()).unwrap_or(5_000);

            let dataset = FaceDataset::load(Path::new("data/faces"), IMAGE_SIZE as u32)?;
            println!("Loaded {} images", dataset.len());

            let param_count: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
            println!("UNet parameter count: {param_count}");

            if Path::new(CHECKPOINT_PATH).exists() {
                varmap.load(CHECKPOINT_PATH)?;
                println!(
                    "Resuming from checkpoint at {CHECKPOINT_PATH} \
                     (note: AdamW's momentum/variance state is not saved, \
                     so the optimizer restarts fresh even though weights don't)"
                );
            }

            let ema = Ema::new(&varmap, &ema_varmap, EMA_DECAY)?;
            if Path::new(EMA_CHECKPOINT_PATH).exists() {
                ema_varmap.load(EMA_CHECKPOINT_PATH)?;
                println!("Resuming EMA from {EMA_CHECKPOINT_PATH}");
            } else {
                // Start the average from the current weights (a resumed
                // checkpoint, or random init) rather than from the shadow
                // model's own unrelated random init.
                ema.copy_from_model()?;
                println!("No EMA checkpoint found - initializing EMA from current weights");
            }

            let config = TrainConfig {
                steps,
                batch_size: 64,
                lr: 2e-4,
                log_every: 50,
                save_every: 500,
                checkpoint_path: CHECKPOINT_PATH.to_string(),
                ema_checkpoint_path: EMA_CHECKPOINT_PATH.to_string(),
            };
            train::train(
                &unet,
                &schedule,
                &dataset,
                &varmap,
                &ema,
                &ema_varmap,
                &device,
                &config,
            )?;
        }
    }

    Ok(())
}
