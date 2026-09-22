use candle_core::{Device, Result, Tensor};
use candle_nn::{
    conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder,
};

const NUM_GROUPS: usize = 8;

/// Fixed (non-learned) sinusoidal embedding of the timestep, the same idea
/// as Transformer positional encodings. `dim` must be even.
fn sinusoidal_embedding(t: &[usize], dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut data = vec![0f32; t.len() * dim];
    for (i, &ti) in t.iter().enumerate() {
        for j in 0..half {
            let freq = (-(10000f32.ln()) * j as f32 / (half as f32 - 1.0)).exp();
            let angle = ti as f32 * freq;
            data[i * dim + j] = angle.sin();
            data[i * dim + half + j] = angle.cos();
        }
    }
    Tensor::from_vec(data, (t.len(), dim), device)
}

/// Learned MLP that turns the fixed sinusoidal embedding into a richer
/// per-sample "what noise level am I at" vector, shared across every block.
struct TimeMlp {
    lin1: Linear,
    lin2: Linear,
}

impl TimeMlp {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: linear(dim, dim, vb.pp("lin1"))?,
            lin2: linear(dim, dim, vb.pp("lin2"))?,
        })
    }

    fn forward(&self, emb: &Tensor) -> Result<Tensor> {
        self.lin2.forward(&self.lin1.forward(emb)?.silu()?)
    }
}

/// The core repeated block: GroupNorm+SiLU+Conv twice, with the timestep
/// embedding injected as a per-channel bias after the first conv, and a
/// residual connection around the whole thing (via a 1x1 conv when the
/// channel count changes, so the shapes match for the add).
struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_proj: Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
    skip: Option<Conv2d>,
}

impl ResBlock {
    fn new(in_ch: usize, out_ch: usize, time_dim: usize, vb: VarBuilder) -> Result<Self> {
        let pad1 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let skip = if in_ch != out_ch {
            Some(conv2d(in_ch, out_ch, 1, Conv2dConfig::default(), vb.pp("skip"))?)
        } else {
            None
        };
        Ok(Self {
            norm1: group_norm(NUM_GROUPS, in_ch, 1e-5, vb.pp("norm1"))?,
            conv1: conv2d(in_ch, out_ch, 3, pad1, vb.pp("conv1"))?,
            time_proj: linear(time_dim, out_ch, vb.pp("time_proj"))?,
            norm2: group_norm(NUM_GROUPS, out_ch, 1e-5, vb.pp("norm2"))?,
            conv2: conv2d(out_ch, out_ch, 3, pad1, vb.pp("conv2"))?,
            skip,
        })
    }

    fn forward(&self, x: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;

        let t = self.time_proj.forward(time_emb)?;
        let dims = t.dims();
        let t = t.reshape((dims[0], dims[1], 1, 1))?;
        let h = h.broadcast_add(&t)?;

        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;

        let residual = match &self.skip {
            Some(skip) => skip.forward(x)?,
            None => x.clone(),
        };
        h + residual
    }
}

/// Strided conv: halves spatial resolution, changes channel count.
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(in_ch, out_ch, 3, cfg, vb.pp("conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Nearest-neighbor upsample followed by a conv (rather than a transposed
/// conv) to avoid checkerboard artifacts; also changes channel count.
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let pad1 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(in_ch, out_ch, 3, pad1, vb.pp("conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (h, w) = (dims[2], dims[3]);
        self.conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)
    }
}

/// A small 2-level U-Net: 64x64 -> 32x32 -> 16x16 and back, with skip
/// connections carrying fine detail across the bottleneck.
pub struct UNet {
    time_dim: usize,
    time_mlp: TimeMlp,
    init_conv: Conv2d,
    enc1: ResBlock,
    down1: Downsample,
    enc2: ResBlock,
    down2: Downsample,
    bott1: ResBlock,
    bott2: ResBlock,
    dec2_up: Upsample,
    dec2: ResBlock,
    dec1_up: Upsample,
    dec1: ResBlock,
    out_norm: GroupNorm,
    out_conv: Conv2d,
}

impl UNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let base = 32;
        let time_dim = base * 4;
        let pad1 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        Ok(Self {
            time_dim,
            time_mlp: TimeMlp::new(time_dim, vb.pp("time_mlp"))?,
            init_conv: conv2d(3, base, 3, pad1, vb.pp("init_conv"))?,

            enc1: ResBlock::new(base, base, time_dim, vb.pp("enc1"))?,
            down1: Downsample::new(base, base * 2, vb.pp("down1"))?,

            enc2: ResBlock::new(base * 2, base * 2, time_dim, vb.pp("enc2"))?,
            down2: Downsample::new(base * 2, base * 4, vb.pp("down2"))?,

            bott1: ResBlock::new(base * 4, base * 4, time_dim, vb.pp("bott1"))?,
            bott2: ResBlock::new(base * 4, base * 4, time_dim, vb.pp("bott2"))?,

            dec2_up: Upsample::new(base * 4, base * 2, vb.pp("dec2_up"))?,
            dec2: ResBlock::new(base * 4, base * 2, time_dim, vb.pp("dec2"))?,

            dec1_up: Upsample::new(base * 2, base, vb.pp("dec1_up"))?,
            dec1: ResBlock::new(base * 2, base, time_dim, vb.pp("dec1"))?,

            out_norm: group_norm(NUM_GROUPS, base, 1e-5, vb.pp("out_norm"))?,
            out_conv: conv2d(base, 3, 3, pad1, vb.pp("out_conv"))?,
        })
    }

    /// Predict the noise added to `x`, given the noisy image and one
    /// timestep index per sample in the batch.
    pub fn forward(&self, x: &Tensor, t: &[usize]) -> Result<Tensor> {
        let emb = sinusoidal_embedding(t, self.time_dim, x.device())?;
        let emb = self.time_mlp.forward(&emb)?;

        let x = self.init_conv.forward(x)?;
        let skip1 = self.enc1.forward(&x, &emb)?;
        let x = self.down1.forward(&skip1)?;

        let skip2 = self.enc2.forward(&x, &emb)?;
        let x = self.down2.forward(&skip2)?;

        let x = self.bott1.forward(&x, &emb)?;
        let x = self.bott2.forward(&x, &emb)?;

        let x = self.dec2_up.forward(&x)?;
        let x = Tensor::cat(&[&x, &skip2], 1)?;
        let x = self.dec2.forward(&x, &emb)?;

        let x = self.dec1_up.forward(&x)?;
        let x = Tensor::cat(&[&x, &skip1], 1)?;
        let x = self.dec1.forward(&x, &emb)?;

        self.out_conv.forward(&self.out_norm.forward(&x)?.silu()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn forward_preserves_input_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let unet = UNet::new(vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &device).unwrap();
        let t = [0usize, 200];

        let out = unet.forward(&x, &t).unwrap();
        assert_eq!(out.dims(), &[2, 3, 64, 64]);
    }
}
