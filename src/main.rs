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
use unet::{UNet, UNetConfig};

const IMAGE_SIZE: usize = 64;
const TIMESTEPS: usize = 400;
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

    // `--large` anywhere on the command line selects the bigger model, which
    // has its own checkpoint and sample files so both sizes can coexist
    // (their checkpoints are not interchangeable).
    let all_args: Vec<String> = std::env::args().skip(1).collect();
    let large = all_args.iter().any(|a| a == "--large");
    let (config, checkpoint_path, ema_checkpoint_path, sample_path) = if large {
        (
            UNetConfig::large(),
            "checkpoints/unet_large.safetensors",
            "checkpoints/unet_large_ema.safetensors",
            "samples/grid_large.png",
        )
    } else {
        (
            UNetConfig::small(),
            "checkpoints/unet.safetensors",
            "checkpoints/unet_ema.safetensors",
            "samples/grid.png",
        )
    };
    println!("Model: {}", if large { "large" } else { "small" });

    let schedule = NoiseSchedule::new(TIMESTEPS);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let unet = UNet::new(&config, vb)?;

    // A second, identically-shaped U-Net holding the EMA of the weights
    // above. Never trained directly; `Ema::update` blends it toward `unet`.
    let mut ema_varmap = VarMap::new();
    let ema_vb = VarBuilder::from_varmap(&ema_varmap, DType::F32, &device);
    let ema_unet = UNet::new(&config, ema_vb)?;

    let mut args = all_args.into_iter().filter(|a| a != "--large");
    match args.next().as_deref() {
        Some("sample") => {
            let batch = args
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);
            let raw = args.next().as_deref() == Some("raw");

            // Prefer the EMA weights; `sample <n> raw` uses the live
            // training weights instead, for comparison.
            let model = if !raw && Path::new(ema_checkpoint_path).exists() {
                ema_varmap.load(ema_checkpoint_path)?;
                println!("Loaded EMA checkpoint from {ema_checkpoint_path}");
                &ema_unet
            } else {
                varmap.load(checkpoint_path)?;
                println!("Loaded checkpoint from {checkpoint_path}");
                &unet
            };

            let images = sample::sample(model, &schedule, batch, IMAGE_SIZE, &device)?;

            let out_path = Path::new(sample_path);
            sample::save_grid(&images, out_path, 4)?;
            println!("Saved {batch} samples to {}", out_path.display());
        }
        other => {
            let steps = other.and_then(|s| s.parse::<usize>().ok()).unwrap_or(5_000);

            let dataset = FaceDataset::load(Path::new("data/faces"), IMAGE_SIZE as u32)?;
            println!("Loaded {} images", dataset.len());

            let param_count: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
            println!("UNet parameter count: {param_count}");

            if Path::new(checkpoint_path).exists() {
                varmap.load(checkpoint_path)?;
                println!(
                    "Resuming from checkpoint at {checkpoint_path} \
                     (note: AdamW's momentum/variance state is not saved, \
                     so the optimizer restarts fresh even though weights don't)"
                );
            }

            let ema = Ema::new(&varmap, &ema_varmap, EMA_DECAY)?;
            if Path::new(ema_checkpoint_path).exists() {
                ema_varmap.load(ema_checkpoint_path)?;
                println!("Resuming EMA from {ema_checkpoint_path}");
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
                checkpoint_path: checkpoint_path.to_string(),
                ema_checkpoint_path: ema_checkpoint_path.to_string(),
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
