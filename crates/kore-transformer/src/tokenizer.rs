//! HuggingFace tokenizer integration.
//!
//! Wraps the `tokenizers` crate to provide encode/decode for Kore models.
//! Loads `tokenizer.json` from a model directory.

use std::path::Path;
use kore_core::KoreError;

/// Wrapper around a HuggingFace tokenizer.
pub struct KoreTokenizer {
    inner: tokenizers::Tokenizer,
}

impl KoreTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, KoreError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| KoreError::StorageError(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a model directory (looks for `tokenizer.json`).
    pub fn from_dir(dir: &Path) -> Result<Self, KoreError> {
        let path = dir.join("tokenizer.json");
        if !path.exists() {
            return Err(KoreError::StorageError(format!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        }
        Self::from_file(&path)
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, KoreError> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| KoreError::StorageError(format!("Tokenizer encode error: {}", e)))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Encode text with special tokens (e.g., BOS/EOS).
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<usize>, KoreError> {
        let encoding = self.inner.encode(text, true)
            .map_err(|e| KoreError::StorageError(format!("Tokenizer encode error: {}", e)))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[usize]) -> Result<String, KoreError> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&ids_u32, true)
            .map_err(|e| KoreError::StorageError(format!("Tokenizer decode error: {}", e)))
    }

    /// Decode token IDs without skipping special tokens.
    pub fn decode_raw(&self, ids: &[usize]) -> Result<String, KoreError> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&ids_u32, false)
            .map_err(|e| KoreError::StorageError(format!("Tokenizer decode error: {}", e)))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Look up the token ID for a string. Returns None if not found.
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.inner.token_to_id(token).map(|id| id as usize)
    }

    /// Look up the string for a token ID. Returns None if not found.
    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.inner.id_to_token(id as u32)
    }

    /// Get the EOS token ID if the tokenizer defines one.
    /// Checks common EOS token strings used by various models.
    pub fn eos_token_id(&self) -> Option<usize> {
        let eos = "<" .to_owned() + "/s>";
        let eot = "<" .to_owned() + "|endoftext|>";
        let eim = "<" .to_owned() + "|end|>";
        self.token_to_id(&eos)
            .or_else(|| self.token_to_id(&eot))
            .or_else(|| self.token_to_id(&eim))
    }

    /// Get the BOS token ID if the tokenizer defines one.
    pub fn bos_token_id(&self) -> Option<usize> {
        let bos = "<" .to_owned() + "s>";
        let bot = "<" .to_owned() + "|begin_of_text|>";
        self.token_to_id(&bos)
            .or_else(|| self.token_to_id(&bot))
    }
}
