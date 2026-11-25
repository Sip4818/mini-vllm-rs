use candle_core::{Result, Tensor, D};

pub struct RmsNorm {
    weights: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weights: Tensor, eps: f64) -> Self {
        Self {
            weights,
            eps,
        }
    }
    pub fn forward(&self, token_embeddings: &Tensor) -> Result<Tensor> {

        let msq = token_embeddings
                    .sqr()?
                    .mean_keepdim(D::Minus1)?;
        let rms = (msq + self.eps)?.sqrt()?;

        token_embeddings.broadcast_div(&rms)?.broadcast_mul(&self.weights)


    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_new() -> candle_core::Result<()>{
        let weights: Tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let eps: f64 = 1e-5_f64;
        let norm = RmsNorm::new(weights.clone(), eps);
        assert_eq!(norm.weights.shape(), weights.shape());
        assert_eq!(norm.weights.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(norm.eps, eps);
        Ok(())
    }

    #[test]
    fn test_forward() -> candle_core::Result<()> {
        let weights = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let eps: f64 = 1e-5_f64;
        let norm = RmsNorm::new(weights, eps);

        let token_embeddings = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [2.0f32, 2.0, 2.0, 2.0]],
            &Device::Cpu,
        )?;
        let normalized_tensor = norm.forward(&token_embeddings)?;

        let expected = Tensor::new(
            &[
                [0.365148f32, 1.460593, 3.286335, 5.842371],
                [1.0f32, 2.0, 3.0, 4.0],
            ],
            &Device::Cpu,
        )?;

        assert_eq!(normalized_tensor.shape(), expected.shape());

        let max_diff = normalized_tensor
            .sub(&expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(max_diff < 1e-5, "max_diff was {max_diff}");

        Ok(())
    }
}