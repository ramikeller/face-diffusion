mod data;
mod schedule;
mod unet;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use data::FaceDataset;
use std::path::Path;
use unet::UNet;

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Using device: {device:?}");

    let dataset = FaceDataset::load(Path::new("data/faces"), 64)?;
    println!("Loaded {} images", dataset.len());

    let batch = dataset.random_batch(8, &device)?;
    println!("Batch shape: {:?}", batch.shape());
    println!(
        "Batch min/max: {:?} / {:?}",
        batch.min_all()?.to_scalar::<f32>()?,
        batch.max_all()?.to_scalar::<f32>()?
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let unet = UNet::new(vb)?;

    let param_count: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("UNet parameter count: {param_count}");

    let t = vec![100usize; 8];
    let predicted_noise = unet.forward(&batch, &t)?;
    println!("Predicted noise shape: {:?}", predicted_noise.shape());

    Ok(())
}
