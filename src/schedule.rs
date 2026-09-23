use anyhow::Result;
use candle_core::Tensor;

/// Precomputed constants for the fixed forward (noising) process and its
/// learned reverse (denoising) process.
pub struct NoiseSchedule {
    pub timesteps: usize,
    sqrt_alphas_cumprod: Vec<f32>,
    sqrt_one_minus_alphas_cumprod: Vec<f32>,
    betas: Vec<f32>,
    sqrt_recip_alphas: Vec<f32>,
}

impl NoiseSchedule {
    /// Linear beta schedule, as in the original DDPM paper (which used
    /// T=1000). The bounds are scaled by 1000/T so the total accumulated
    /// noise stays the same regardless of step count: without this, a
    /// smaller T leaves meaningful signal in x_T, mismatching the pure-noise
    /// starting point used at sampling time.
    pub fn new(timesteps: usize) -> Self {
        let scale = 1000.0 / timesteps as f32;
        let beta_start = 1e-4_f32 * scale;
        let beta_end = 0.02_f32 * scale;
        let betas: Vec<f32> = (0..timesteps)
            .map(|i| beta_start + (beta_end - beta_start) * i as f32 / (timesteps - 1) as f32)
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut running = 1.0_f32;
        for &beta in &betas {
            running *= 1.0 - beta;
            alphas_cumprod.push(running);
        }

        let sqrt_alphas_cumprod = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod =
            alphas_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();
        let sqrt_recip_alphas = betas.iter().map(|b| (1.0 - b).sqrt().recip()).collect();

        Self {
            timesteps,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            betas,
            sqrt_recip_alphas,
        }
    }

    /// x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    ///
    /// `t` holds one timestep index per sample in the batch; `noise` must be
    /// the same shape as `x0`.
    pub fn q_sample(&self, x0: &Tensor, t: &[usize], noise: &Tensor) -> Result<Tensor> {
        let batch = t.len();
        let device = x0.device();

        let sqrt_ac: Vec<f32> = t.iter().map(|&ti| self.sqrt_alphas_cumprod[ti]).collect();
        let sqrt_omac: Vec<f32> = t
            .iter()
            .map(|&ti| self.sqrt_one_minus_alphas_cumprod[ti])
            .collect();

        // Reshape to (batch, 1, 1, 1) so it broadcasts against (batch, C, H, W).
        let sqrt_ac = Tensor::from_vec(sqrt_ac, (batch, 1, 1, 1), device)?;
        let sqrt_omac = Tensor::from_vec(sqrt_omac, (batch, 1, 1, 1), device)?;

        let signal = x0.broadcast_mul(&sqrt_ac)?;
        let noise_term = noise.broadcast_mul(&sqrt_omac)?;
        Ok((signal + noise_term)?)
    }

    /// One reverse-diffusion step: x_t -> x_{t-1}, given the model's noise
    /// prediction at x_t. Unlike `q_sample`, `t` is a single shared timestep
    /// for the whole batch — sampling advances every sample in lockstep,
    /// there's no per-sample randomness in *which* step we're on.
    pub fn p_sample(
        &self,
        xt: &Tensor,
        predicted_noise: &Tensor,
        t: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let coeff = self.betas[t] as f64 / self.sqrt_one_minus_alphas_cumprod[t] as f64;
        let scaled_noise_pred = (predicted_noise * coeff)?;
        let mean = ((xt - scaled_noise_pred)? * self.sqrt_recip_alphas[t] as f64)?;

        if t == 0 {
            // No next step to feed a distribution into: output the model's
            // best estimate directly instead of injecting more randomness.
            Ok(mean)
        } else {
            let sigma = (self.betas[t] as f64).sqrt();
            Ok((mean + (noise * sigma)?)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_endpoints_are_sane() {
        let schedule = NoiseSchedule::new(400);
        // Barely noised at t=0: almost all signal survives.
        assert!(schedule.sqrt_alphas_cumprod[0] > 0.99);
        // Almost pure noise by the final step.
        assert!(schedule.sqrt_alphas_cumprod[399] < 0.01);
        // The two coefficients always form a unit vector (variance-preserving).
        for i in [0, 199, 399] {
            let a = schedule.sqrt_alphas_cumprod[i];
            let b = schedule.sqrt_one_minus_alphas_cumprod[i];
            assert!((a * a + b * b - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn q_sample_produces_expected_shape() {
        let schedule = NoiseSchedule::new(400);
        let device = candle_core::Device::Cpu;
        let x0 = Tensor::zeros((4, 3, 64, 64), candle_core::DType::F32, &device).unwrap();
        let noise = Tensor::ones((4, 3, 64, 64), candle_core::DType::F32, &device).unwrap();
        let t = [0usize, 100, 200, 399];

        let xt = schedule.q_sample(&x0, &t, &noise).unwrap();
        assert_eq!(xt.dims(), &[4, 3, 64, 64]);
    }

    #[test]
    fn p_sample_produces_expected_shape_and_is_deterministic_at_t0() {
        let schedule = NoiseSchedule::new(400);
        let device = candle_core::Device::Cpu;
        let xt = Tensor::zeros((2, 3, 64, 64), candle_core::DType::F32, &device).unwrap();
        let predicted_noise = Tensor::zeros((2, 3, 64, 64), candle_core::DType::F32, &device).unwrap();
        let noise = Tensor::ones((2, 3, 64, 64), candle_core::DType::F32, &device).unwrap();

        let x_mid = schedule.p_sample(&xt, &predicted_noise, 200, &noise).unwrap();
        assert_eq!(x_mid.dims(), &[2, 3, 64, 64]);

        // At t=0 no fresh noise is injected, so the result must not depend
        // on `noise` at all - swap it for something else and confirm the
        // output is unchanged.
        let other_noise = (&noise * 5.0).unwrap();
        let x0_a = schedule.p_sample(&xt, &predicted_noise, 0, &noise).unwrap();
        let x0_b = schedule.p_sample(&xt, &predicted_noise, 0, &other_noise).unwrap();
        let diff = (x0_a - x0_b).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0);
    }
}
