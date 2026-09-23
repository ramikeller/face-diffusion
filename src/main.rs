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

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Using device: {device:?}");

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

            let config = TrainConfig {
                steps,
                batch_size: 64,
                lr: 2e-4,
                log_every: 50,
                checkpoint_path: CHECKPOINT_PATH.to_string(),
            };
            train::train(&unet, &schedule, &dataset, &varmap, &device, &config)?;
        }
    }

    Ok(())
}
