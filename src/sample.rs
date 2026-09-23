use crate::schedule::NoiseSchedule;
use crate::unet::UNet;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use std::path::Path;

/// Runs the full reverse process: pure Gaussian noise -> a generated image
/// batch, still in the network's [-1, 1] working range. `T` sequential
/// forward passes through the U-Net, one per timestep from T-1 down to 0.
pub fn sample(
    unet: &UNet,
    schedule: &NoiseSchedule,
    batch: usize,
    image_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut x = Tensor::randn(0f32, 1f32, (batch, 3, image_size, image_size), device)?;

    for t in (0..schedule.timesteps).rev() {
        let t_batch = vec![t; batch];
        let predicted_noise = unet.forward(&x, &t_batch)?;
        let noise = x.randn_like(0.0, 1.0)?;
        // The U-Net's weights are trainable Vars, so every forward pass
        // still builds an autodiff graph even though we never call
        // .backward() here. Left attached, x's graph would keep growing
        // across all T sequential steps (each step's output feeds the
        // next), retaining every prior step's activations until memory is
        // exhausted. detach() severs it each step since sampling has no use
        // for gradients at all.
        x = schedule.p_sample(&x, &predicted_noise, t, &noise)?.detach();

        if t % 50 == 0 {
            println!("sample step {}/{}", schedule.timesteps - t, schedule.timesteps);
        }
    }

    Ok(x)
}

/// Lays a batch of images (n, 3, h, w) in [-1, 1] out as a grid PNG, `cols`
/// images per row.
pub fn save_grid(images: &Tensor, path: &Path, cols: usize) -> Result<()> {
    let images = images.clamp(-1f64, 1f64)?.affine(127.5, 127.5)?.to_dtype(DType::U8)?;
    let dims = images.dims();
    let (batch, h, w) = (dims[0], dims[2], dims[3]);
    let rows = batch.div_ceil(cols);

    let mut canvas = image::RgbImage::new((cols * w) as u32, (rows * h) as u32);

    for i in 0..batch {
        let pixels = images.i(i)?.permute((1, 2, 0))?.to_vec3::<u8>()?;
        let (row, col) = (i / cols, i % cols);
        for (y, row_pixels) in pixels.iter().enumerate() {
            for (x, p) in row_pixels.iter().enumerate() {
                canvas.put_pixel(
                    (col * w + x) as u32,
                    (row * h + y) as u32,
                    image::Rgb([p[0], p[1], p[2]]),
                );
            }
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    canvas.save(path)?;
    Ok(())
}
