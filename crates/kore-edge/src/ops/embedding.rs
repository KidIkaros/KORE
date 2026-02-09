//! Embedding lookup on raw slices.

/// Look up embeddings for token IDs.
///
/// `weight`: [vocab_size, dim] embedding table
/// `ids`: token IDs to look up
/// `output`: [ids.len(), dim] output buffer
pub fn embedding_lookup(weight: &[f32], ids: &[u32], output: &mut [f32], vocab_size: usize, dim: usize) {
    for (i, &id) in ids.iter().enumerate() {
        let id = id as usize;
        let dst = &mut output[i * dim..(i + 1) * dim];
        if id < vocab_size {
            let src = &weight[id * dim..(id + 1) * dim];
            dst.copy_from_slice(src);
        } else {
            for v in dst.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        // vocab=3, dim=2
        let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ids = [0u32, 2, 1];
        let mut output = [0.0f32; 6];
        embedding_lookup(&weight, &ids, &mut output, 3, 2);
        assert_eq!(output, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embedding_out_of_range() {
        let weight = [1.0, 2.0, 3.0, 4.0];
        let ids = [10u32]; // out of range
        let mut output = [99.0f32; 2];
        embedding_lookup(&weight, &ids, &mut output, 2, 2);
        assert_eq!(output, [0.0, 0.0]);
    }
}
