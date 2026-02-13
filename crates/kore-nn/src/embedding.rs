//! Embedding layer — lookup table for token IDs to dense vectors.

use kore_core::Tensor;
use crate::module::Module;

/// Embedding lookup table: maps integer token IDs to dense vectors.
#[derive(Clone)]
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    training: bool,
}

impl Embedding {
    /// Create a new Embedding layer with random initialization.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Normal(0, 0.02) initialization (standard for embeddings)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total = num_embeddings * embedding_dim;
        let data: Vec<f32> = (0..total)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rng.gen_range(1e-7..1.0);
                let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                (-2.0 * u1.ln()).sqrt() * u2.cos() * 0.02
            })
            .collect();

        let mut weight = Tensor::from_f32(&data, &[num_embeddings, embedding_dim]);
        weight.set_requires_grad(true);

        Self {
            weight,
            num_embeddings,
            embedding_dim,
            training: true,
        }
    }

    /// Create from an existing weight tensor [num_embeddings, embedding_dim].
    pub fn from_weight(weight: Tensor) -> Self {
        let dims = weight.shape().dims().to_vec();
        assert_eq!(dims.len(), 2, "Embedding weight must be 2D");
        Self {
            num_embeddings: dims[0],
            embedding_dim: dims[1],
            weight,
            training: true,
        }
    }

    /// Look up embeddings for a slice of token IDs.
    /// Returns tensor of shape [ids.len(), embedding_dim].
    pub fn lookup(&self, ids: &[usize]) -> Tensor {
        let w = self.weight.contiguous();
        let w_data = w.as_f32_slice()
            .expect("Embedding: weight tensor must be F32");
        let dim = self.embedding_dim;

        let mut result = vec![0.0f32; ids.len() * dim];
        for (i, &id) in ids.iter().enumerate() {
            if id < self.num_embeddings {
                let src = &w_data[id * dim..(id + 1) * dim];
                result[i * dim..(i + 1) * dim].copy_from_slice(src);
            }
            // Out-of-range IDs get zero vectors (already initialized)
        }

        Tensor::from_f32(&result, &[ids.len(), dim])
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Number of embeddings.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Module for Embedding {
    fn clone_box(&self) -> Box<dyn Module> { Box::new(self.clone()) }

    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        // Input is expected to be a 1D tensor of token IDs (as f32, cast to usize)
        let data = input.contiguous();
        let slice = data.as_f32_slice().ok_or_else(|| {
            kore_core::KoreError::UnsupportedDType(input.dtype())
        })?;
        let ids: Vec<usize> = slice.iter().map(|&v| v as usize).collect();
        Ok(self.lookup(&ids))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        vec![("weight".into(), &self.weight)]
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        self.weight = params[0].clone();
        1
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let emb = Embedding::new(100, 32);
        assert_eq!(emb.num_embeddings(), 100);
        assert_eq!(emb.embedding_dim(), 32);
        assert_eq!(emb.weight().shape().dims(), &[100, 32]);
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = Embedding::new(10, 4);
        let result = emb.lookup(&[0, 5, 9]);
        assert_eq!(result.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_embedding_lookup_consistency() {
        let emb = Embedding::new(10, 4);
        let r1 = emb.lookup(&[3]);
        let r2 = emb.lookup(&[3]);
        assert_eq!(r1.as_f32_slice().unwrap(), r2.as_f32_slice().unwrap());
    }

    #[test]
    fn test_embedding_out_of_range() {
        let emb = Embedding::new(5, 3);
        let result = emb.lookup(&[10]); // out of range
        let data = result.as_f32_slice().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embedding_forward() {
        let emb = Embedding::new(10, 4);
        let input = Tensor::from_f32(&[0.0, 3.0, 7.0], &[3]);
        let output = emb.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_embedding_parameters() {
        let emb = Embedding::new(50, 16);
        assert_eq!(emb.parameters().len(), 1);
    }
}
