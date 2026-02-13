//! DataLoader — PyTorch-style batched iteration over datasets.
//!
//! Provides a general-purpose `DataLoader` that wraps any `Dataset`
//! implementation and yields `(input, target)` Tensor batches.

use kore_core::Tensor;
use rand::seq::SliceRandom;

/// A single sample: `(input, target)` as raw f32 vectors.
///
/// This is the unit that `Dataset` implementations return.
#[derive(Debug, Clone)]
pub struct Sample {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

/// Trait for indexable datasets.
///
/// Implement this to plug any data source into `DataLoader`.
pub trait Dataset: Send + Sync {
    /// Total number of samples.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single sample by index.
    fn get(&self, index: usize) -> Sample;

    /// Input feature dimension.
    fn input_dim(&self) -> usize;

    /// Target dimension.
    fn target_dim(&self) -> usize;
}

/// A simple in-memory dataset of `(input, target)` pairs stored as Tensors.
pub struct TensorDataset {
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    input_dim: usize,
    target_dim: usize,
}

impl TensorDataset {
    /// Create a TensorDataset from input and target Tensors.
    ///
    /// Both tensors must have the same first dimension (number of samples).
    /// - `inputs`: `[N, input_dim]`
    /// - `targets`: `[N, target_dim]`
    pub fn new(inputs: &Tensor, targets: &Tensor) -> Self {
        let in_dims = inputs.shape().dims();
        let tgt_dims = targets.shape().dims();
        assert_eq!(in_dims[0], tgt_dims[0], "inputs and targets must have same number of samples");

        let n = in_dims[0];
        let input_dim = if in_dims.len() > 1 { in_dims[1] } else { 1 };
        let target_dim = if tgt_dims.len() > 1 { tgt_dims[1] } else { 1 };

        let in_data = inputs.as_f32_slice().expect("inputs must be f32");
        let tgt_data = targets.as_f32_slice().expect("targets must be f32");

        let mut in_vecs = Vec::with_capacity(n);
        let mut tgt_vecs = Vec::with_capacity(n);
        for i in 0..n {
            in_vecs.push(in_data[i * input_dim..(i + 1) * input_dim].to_vec());
            tgt_vecs.push(tgt_data[i * target_dim..(i + 1) * target_dim].to_vec());
        }

        Self {
            inputs: in_vecs,
            targets: tgt_vecs,
            input_dim,
            target_dim,
        }
    }

    /// Create from raw Vec pairs.
    pub fn from_vecs(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        assert_eq!(inputs.len(), targets.len(), "inputs and targets must have same length");
        let input_dim = inputs.first().map(|v| v.len()).unwrap_or(0);
        let target_dim = targets.first().map(|v| v.len()).unwrap_or(0);
        Self { inputs, targets, input_dim, target_dim }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> Sample {
        Sample {
            input: self.inputs[index].clone(),
            target: self.targets[index].clone(),
        }
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn target_dim(&self) -> usize {
        self.target_dim
    }
}

/// A batch of `(input, target)` Tensors.
#[derive(Debug)]
pub struct Batch {
    /// Input tensor of shape `[batch_size, input_dim]`.
    pub input: Tensor,
    /// Target tensor of shape `[batch_size, target_dim]`.
    pub target: Tensor,
    /// Number of samples in this batch.
    pub size: usize,
}

/// PyTorch-style DataLoader for batched iteration over a `Dataset`.
///
/// # Example (Rust)
/// ```ignore
/// use kore_data::{DataLoader, TensorDataset};
/// use kore_core::Tensor;
///
/// let x = Tensor::randn(&[100, 16]);
/// let y = Tensor::randn(&[100, 1]);
/// let ds = TensorDataset::new(&x, &y);
/// let loader = DataLoader::new(Box::new(ds), 32, true, true, Some(42));
///
/// for batch in loader.iter() {
///     // batch.input: [32, 16], batch.target: [32, 1]
/// }
/// ```
pub struct DataLoader {
    dataset: Box<dyn Dataset>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: Option<u64>,
    epoch_counter: std::cell::Cell<u64>,
}

impl DataLoader {
    /// Create a new DataLoader.
    ///
    /// - `dataset`: the underlying dataset
    /// - `batch_size`: samples per batch
    /// - `shuffle`: whether to shuffle indices each epoch
    /// - `drop_last`: if true, drop the last incomplete batch
    /// - `seed`: optional RNG seed for reproducible shuffling
    pub fn new(
        dataset: Box<dyn Dataset>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        seed: Option<u64>,
    ) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self { dataset, batch_size, shuffle, drop_last, seed, epoch_counter: std::cell::Cell::new(0) }
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Total number of samples.
    pub fn num_samples(&self) -> usize {
        self.dataset.len()
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Current epoch counter (incremented each `iter()` call).
    pub fn epoch(&self) -> u64 {
        self.epoch_counter.get()
    }

    /// Set the epoch counter (useful when resuming training).
    pub fn set_epoch(&self, epoch: u64) {
        self.epoch_counter.set(epoch);
    }

    /// Return an iterator over batches for one epoch.
    pub fn iter(&self) -> DataLoaderIter<'_> {
        let n = self.dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            use rand::SeedableRng;
            let epoch = self.epoch_counter.get();
            self.epoch_counter.set(epoch + 1);
            if let Some(seed) = self.seed {
                // Vary seed per epoch for different shuffle order while staying deterministic
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(epoch));
                indices.shuffle(&mut rng);
            } else {
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
            }
        }

        DataLoaderIter {
            loader: self,
            indices,
            pos: 0,
        }
    }
}

/// Iterator over DataLoader batches.
pub struct DataLoaderIter<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    pos: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.indices.len();
        if self.pos >= n {
            return None;
        }

        let end = (self.pos + self.loader.batch_size).min(n);
        let batch_indices = &self.indices[self.pos..end];
        let actual_batch_size = batch_indices.len();

        // Drop the last incomplete batch if requested
        if self.loader.drop_last && actual_batch_size < self.loader.batch_size {
            return None;
        }

        let input_dim = self.loader.dataset.input_dim();
        let target_dim = self.loader.dataset.target_dim();

        let mut input_data = Vec::with_capacity(actual_batch_size * input_dim);
        let mut target_data = Vec::with_capacity(actual_batch_size * target_dim);

        for &idx in batch_indices {
            let sample = self.loader.dataset.get(idx);
            input_data.extend_from_slice(&sample.input);
            target_data.extend_from_slice(&sample.target);
        }

        let input = Tensor::from_f32(&input_data, &[actual_batch_size, input_dim]);
        let target = Tensor::from_f32(&target_data, &[actual_batch_size, target_dim]);

        self.pos = end;

        Some(Batch {
            input,
            target,
            size: actual_batch_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_core::DType;

    fn make_dataset(n: usize, in_dim: usize, tgt_dim: usize) -> TensorDataset {
        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..in_dim).map(|j| (i * in_dim + j) as f32).collect())
            .collect();
        let targets: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..tgt_dim).map(|j| (i * tgt_dim + j) as f32 * 0.1).collect())
            .collect();
        TensorDataset::from_vecs(inputs, targets)
    }

    #[test]
    fn test_tensor_dataset_from_tensors() {
        let x = Tensor::ones(&[10, 4]);
        let y = Tensor::zeros(&[10, 1], DType::F32);
        let ds = TensorDataset::new(&x, &y);
        assert_eq!(ds.len(), 10);
        assert_eq!(ds.input_dim(), 4);
        assert_eq!(ds.target_dim(), 1);
    }

    #[test]
    fn test_dataloader_basic() {
        let ds = make_dataset(10, 4, 1);
        let loader = DataLoader::new(Box::new(ds), 3, false, false, None);
        assert_eq!(loader.num_batches(), 4); // ceil(10/3) = 4

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].size, 3);
        assert_eq!(batches[3].size, 1); // last incomplete batch
    }

    #[test]
    fn test_dataloader_drop_last() {
        let ds = make_dataset(10, 4, 1);
        let loader = DataLoader::new(Box::new(ds), 3, false, true, None);
        assert_eq!(loader.num_batches(), 3); // floor(10/3) = 3

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.size, 3);
        }
    }

    #[test]
    fn test_dataloader_shuffle_deterministic() {
        // Two loaders with the same seed produce identical first-epoch order
        let ds1 = make_dataset(10, 4, 1);
        let ds2 = make_dataset(10, 4, 1);
        let loader1 = DataLoader::new(Box::new(ds1), 10, true, false, Some(42));
        let loader2 = DataLoader::new(Box::new(ds2), 10, true, false, Some(42));

        let b1: Vec<_> = loader1.iter().collect();
        let b2: Vec<_> = loader2.iter().collect();

        let d1 = b1[0].input.as_f32_slice().unwrap().to_vec();
        let d2 = b2[0].input.as_f32_slice().unwrap().to_vec();
        assert_eq!(d1, d2, "same seed should produce same first-epoch order");
    }

    #[test]
    fn test_dataloader_shuffle_varies_per_epoch() {
        // Successive iter() calls on the same loader produce different order
        let ds = make_dataset(10, 4, 1);
        let loader = DataLoader::new(Box::new(ds), 10, true, false, Some(42));

        let b1: Vec<_> = loader.iter().collect();
        let b2: Vec<_> = loader.iter().collect();

        let d1 = b1[0].input.as_f32_slice().unwrap().to_vec();
        let d2 = b2[0].input.as_f32_slice().unwrap().to_vec();
        assert_ne!(d1, d2, "different epochs should produce different shuffle order");
    }

    #[test]
    fn test_dataloader_shapes() {
        let ds = make_dataset(8, 16, 2);
        let loader = DataLoader::new(Box::new(ds), 4, false, false, None);

        for batch in loader.iter() {
            assert_eq!(batch.input.shape().dims(), &[4, 16]);
            assert_eq!(batch.target.shape().dims(), &[4, 2]);
        }
    }

    #[test]
    fn test_dataloader_exact_divisible() {
        let ds = make_dataset(9, 3, 1);
        let loader = DataLoader::new(Box::new(ds), 3, false, false, None);
        assert_eq!(loader.num_batches(), 3);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.size, 3);
        }
    }

    #[test]
    fn test_dataloader_single_sample() {
        let ds = make_dataset(1, 2, 1);
        let loader = DataLoader::new(Box::new(ds), 1, false, false, None);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].input.shape().dims(), &[1, 2]);
    }
}
