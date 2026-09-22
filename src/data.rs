use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::imageops::FilterType;
use rand::RngExt;
use std::path::Path;

/// In-memory face dataset: raw u8 pixels, layout (n, channels, size, size).
pub struct FaceDataset {
    pixels: Tensor,
    len: usize,
}

impl FaceDataset {
    pub fn load(dir: &Path, size: u32) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("jpg" | "jpeg" | "png")
                )
            })
            .collect();
        paths.sort();
        let n = paths.len();
        anyhow::ensure!(n > 0, "no images found in {}", dir.display());

        let mut buffer = Vec::with_capacity(n * 3 * size as usize * size as usize);
        for path in &paths {
            let img = image::open(path)?.into_rgb8();
            let (w, h) = img.dimensions();
            let crop = w.min(h);
            let x0 = (w - crop) / 2;
            let y0 = (h - crop) / 2;
            let cropped = image::imageops::crop_imm(&img, x0, y0, crop, crop).to_image();
            let resized = image::imageops::resize(&cropped, size, size, FilterType::Triangle);

            // HWC (as stored by `image`) -> CHW (as expected by conv layers).
            for c in 0..3 {
                for y in 0..size {
                    for x in 0..size {
                        buffer.push(resized.get_pixel(x, y)[c]);
                    }
                }
            }
        }

        let pixels = Tensor::from_vec(buffer, (n, 3, size as usize, size as usize), &Device::Cpu)?;
        Ok(Self { pixels, len: n })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Sample a random batch, cast to f32 and normalized to [-1, 1], placed on `device`.
    pub fn random_batch(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        let mut rng = rand::rng();
        let idx: Vec<u32> = (0..batch_size)
            .map(|_| rng.random_range(0..self.len as u32))
            .collect();
        let idx = Tensor::from_vec(idx, batch_size, &Device::Cpu)?;
        let batch = self.pixels.index_select(&idx, 0)?.to_device(device)?;
        let batch = batch.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?;
        Ok(batch)
    }
}
