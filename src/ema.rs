use anyhow::{anyhow, Result};
use candle_core::Var;
use candle_nn::VarMap;

/// Exponential moving average of the model's weights. The raw weights
/// jitter step to step with each noisy minibatch gradient, and sampling
/// from them shows that jitter as speckle/blotch artifacts; sampling from a
/// slow running average of them gives much cleaner images from the same
/// model. Standard in DDPM (the original paper uses decay 0.9999).
pub struct Ema {
    decay: f64,
    /// (live training var, its shadow EMA var), paired by name.
    pairs: Vec<(Var, Var)>,
}

impl Ema {
    /// `shadow` must hold the same variable names/shapes as `model`, i.e. a
    /// second `UNet` must have been built from it.
    pub fn new(model: &VarMap, shadow: &VarMap, decay: f64) -> Result<Self> {
        let model_vars = model.data().lock().unwrap();
        let shadow_vars = shadow.data().lock().unwrap();
        let pairs = model_vars
            .iter()
            .map(|(name, var)| {
                let shadow_var = shadow_vars
                    .get(name)
                    .ok_or_else(|| anyhow!("EMA has no variable named {name}"))?;
                Ok((var.clone(), shadow_var.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { decay, pairs })
    }

    /// Overwrites the EMA with the current training weights, e.g. to start
    /// it from an existing checkpoint rather than from random init.
    pub fn copy_from_model(&self) -> Result<()> {
        for (var, shadow) in &self.pairs {
            shadow.set(&var.as_tensor().detach())?;
        }
        Ok(())
    }

    /// shadow = decay * shadow + (1 - decay) * weights
    pub fn update(&self) -> Result<()> {
        for (var, shadow) in &self.pairs {
            let blended = ((shadow.as_tensor() * self.decay)?
                + (var.as_tensor() * (1.0 - self.decay))?)?
                .detach();
            shadow.set(&blended)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn update_blends_toward_model_weights() {
        let device = Device::Cpu;
        let model = VarMap::new();
        let shadow = VarMap::new();
        model.get(4, "w", candle_nn::Init::Const(1.0), DType::F32, &device).unwrap();
        shadow.get(4, "w", candle_nn::Init::Const(0.0), DType::F32, &device).unwrap();

        let ema = Ema::new(&model, &shadow, 0.9).unwrap();
        ema.update().unwrap();

        let w: Tensor = shadow.data().lock().unwrap()["w"].as_tensor().clone();
        for v in w.to_vec1::<f32>().unwrap() {
            assert!((v - 0.1).abs() < 1e-6);
        }

        ema.copy_from_model().unwrap();
        let w: Tensor = shadow.data().lock().unwrap()["w"].as_tensor().clone();
        assert_eq!(w.to_vec1::<f32>().unwrap(), vec![1.0; 4]);
    }
}
