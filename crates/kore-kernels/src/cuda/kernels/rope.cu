// Rotary Position Embedding (RoPE) kernel for Kore.
// Applies rotary embeddings in-place on GPU.
// Each pair of elements (2i, 2i+1) is rotated by angle theta_i * position.

extern "C" {

// Apply RoPE to input tensor.
// input: [seq_len, num_heads, head_dim] (contiguous)
// freqs: [seq_len, head_dim/2] — precomputed frequency values (theta * pos)
// Modifies input in-place.
__global__ void rope_f32(
    float* __restrict__ data,
    const float* __restrict__ freqs,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_dim = head_dim / 2;
    unsigned int total = seq_len * num_heads * half_dim;

    if (idx >= total) return;

    // Decompose flat index into (seq, head, pair)
    unsigned int pair = idx % half_dim;
    unsigned int head = (idx / half_dim) % num_heads;
    unsigned int seq = idx / (half_dim * num_heads);

    // Get the rotation angle for this position and dimension
    float freq = freqs[seq * half_dim + pair];
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    // Indices into the data tensor
    unsigned int base = seq * num_heads * head_dim + head * head_dim;
    unsigned int i0 = base + pair;
    unsigned int i1 = base + pair + half_dim;

    float x0 = data[i0];
    float x1 = data[i1];

    // Apply rotation
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}

// Precompute RoPE frequency table.
// freqs[pos, dim] = pos * theta^(-2*dim/head_dim)
// where theta = 10000.0 (default)
__global__ void rope_freqs_f32(
    float* __restrict__ freqs,
    unsigned int seq_len,
    unsigned int half_dim,
    float theta
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * half_dim;

    if (idx >= total) return;

    unsigned int dim = idx % half_dim;
    unsigned int pos = idx / half_dim;

    float freq = (float)pos / powf(theta, 2.0f * (float)dim / (float)(half_dim * 2));
    freqs[idx] = freq;
}

} // extern "C"
