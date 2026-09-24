use crate::data::FaceDataset;
use crate::schedule::NoiseSchedule;
use crate::unet::UNet;
use anyhow::Result;
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::RngExt;
use std::time::Instant;

pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub log_every: usize,
    pub save_every: usize,
    pub checkpoint_path: String,
}

/// Runs `config.steps` training steps. Each step: a random batch, a random
/// timestep per sample, fresh noise, and one gradient update on how well the
/// U-Net predicted that noise. Saves a checkpoint periodically and when done,
/// so `sample` can be run against progress without waiting for the full run
/// to finish (loss alone is a poor signal for when visual structure
/// emerges - it plateaus early, long before samples look face-like).
pub fn train(
    unet: &UNet,
    schedule: &NoiseSchedule,
    dataset: &FaceDataset,
    varmap: &VarMap,
    device: &Device,
    config: &TrainConfig,
) -> Result<()> {
    let params = ParamsAdamW {
        lr: config.lr,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    if let Some(parent) = std::path::Path::new(&config.checkpoint_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut rng = rand::rng();
    let start = Instant::now();

    for step in 0..config.steps {
        let x0 = dataset.random_batch(config.batch_size, device)?;

        let t: Vec<usize> = (0..config.batch_size)
            .map(|_| rng.random_range(0..schedule.timesteps))
            .collect();

        let noise = x0.randn_like(0.0, 1.0)?;
        let xt = schedule.q_sample(&x0, &t, &noise)?;

        let predicted_noise = unet.forward(&xt, &t)?;
        let loss = candle_nn::loss::mse(&predicted_noise, &noise)?;

        optimizer.backward_step(&loss)?;

        if step % config.log_every == 0 || step == config.steps - 1 {
            let loss_val = loss.to_scalar::<f32>()?;
            let elapsed = start.elapsed().as_secs_f32();
            println!(
                "step {:>6}/{} | loss {loss_val:.4} | {elapsed:7.1}s elapsed | {:.2} steps/s",
                step + 1,
                config.steps,
                (step + 1) as f32 / elapsed.max(0.001)
            );
        }

        if (step + 1) % config.save_every == 0 || step == config.steps - 1 {
            varmap.save(&config.checkpoint_path)?;
            println!("saved checkpoint to {} (step {})", config.checkpoint_path, step + 1);
        }
    }

    Ok(())
}
