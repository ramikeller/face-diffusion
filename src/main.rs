mod data;
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

const TIMESTEPS: usize = 400;

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Using device: {device:?}");

    let dataset = FaceDataset::load(Path::new("data/faces"), 64)?;
    println!("Loaded {} images", dataset.len());

    let schedule = NoiseSchedule::new(TIMESTEPS);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let unet = UNet::new(vb)?;

    let param_count: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("UNet parameter count: {param_count}");

    let steps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5_000);

    let config = TrainConfig {
        steps,
        batch_size: 64,
        lr: 2e-4,
        log_every: 50,
        checkpoint_path: "checkpoints/unet.safetensors".to_string(),
    };
    train::train(&unet, &schedule, &dataset, &varmap, &device, &config)?;

    Ok(())
}
