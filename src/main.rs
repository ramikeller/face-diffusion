mod data;
mod sample;
mod schedule;
mod train;
mod unet;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use data::FaceDataset;
use schedule::NoiseSchedule;
use std::path::Path;
use train::TrainConfig;
use unet::UNet;

const IMAGE_SIZE: usize = 64;
const TIMESTEPS: usize = 400;
const CHECKPOINT_PATH: &str = "checkpoints/unet.safetensors";

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

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("sample") => {
            varmap.load(CHECKPOINT_PATH)?;
            println!("Loaded checkpoint from {CHECKPOINT_PATH}");

            let batch = args
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);

            let images = sample::sample(&unet, &schedule, batch, IMAGE_SIZE, &device)?;

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

            let config = TrainConfig {
                steps,
                batch_size: 64,
                lr: 2e-4,
                log_every: 50,
                save_every: 500,
                checkpoint_path: CHECKPOINT_PATH.to_string(),
            };
            train::train(&unet, &schedule, &dataset, &varmap, &device, &config)?;
        }
    }

    Ok(())
}
