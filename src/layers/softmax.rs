use candle_core:: {Result, Tensor, D};


pub fn softmax(input: &Tensor) -> Result<Tensor> {
    let max = input.max_keepdim(D::Minus1)?;
    let exp_input_stable = input.broadcast_sub(&max)?.exp()?;
    let exp_sum = exp_input_stable.sum_keepdim(D::Minus1)?;
    exp_input_stable.broadcast_div(&exp_sum)
    
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn test_softmax() -> candle_core::Result<()> {
        let device = Device::Cpu;

        let input = Tensor::new(&[2.0f32, 1.0, 0.1], &device)?;

        let output = softmax(&input)?;

        let expected = [0.6590f32, 0.2424, 0.0986];

        let actual = output.to_vec1::<f32>()?;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-3);
        }

        // Probabilities should sum to 1.
        let sum: f32 = actual.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        Ok(())
    }
    #[test]
    fn test_softmax_2d() -> Result<()> {
        let input = Tensor::new(
            &[[1f32, 2., 3.], [1., 2., 3.]],
            &Device::Cpu,
        )?;

        let output = softmax(&input)?;

        let expected = Tensor::new(
            &[
                [0.09003057f32, 0.24472848, 0.66524094],
                [0.09003057, 0.24472848, 0.66524094],
            ],
            &Device::Cpu,
        )?;

        assert_eq!(output.dims(), &[2, 3]);

        let diff = (output - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5);

        Ok(())
    }
}
