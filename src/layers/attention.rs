use candle_core:: {Tensor};

pub struct Attention {
    k_weights: Tensor,
    q_weights: Tensor,
    v_weights: Tensor

}

impl Attention {
    pub fn new(k_weights: Tensor, q_weights Tensor, v_weights: Tensor) -> Self {
        Self {
            k_weights.
            q_weights.
            v_weights
        }
    }

    pub fn forward(&self, token_embeddings: Tensor) -> Result<Tensor>{
        query = token_embeddings.broadcast_matmul(self.q_weights)?;
        key = token_embeddings.broadcast_matmul(self.k_weights)?;
        value = token_embeddings.broadcast_matmul(self.v_weights)?;
        dim = query.dim(-1);
        
        query.broadcast_matmul(&k.t()?)? / (d_k as f32).sqrt())?.broadcast_matmul(value)

    }



}