mod data;

use candle_core::Device;
use data::FaceDataset;
use std::path::Path;

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

    Ok(())
}
