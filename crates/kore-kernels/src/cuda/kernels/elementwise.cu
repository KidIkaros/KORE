// Vectorized element-wise CUDA kernels for Kore.
// Uses float4 loads/stores for coalesced memory access.
// Grid-stride loop pattern handles arbitrary tensor sizes.

extern "C" {

// ============================================================================
// Binary ops: C[i] = op(A[i], B[i])
// ============================================================================

__global__ void add_f32(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    // Vectorized path: process 4 elements at a time
    unsigned int vec_n = n / 4;
    for (unsigned int i = idx; i < vec_n; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }
    // Handle remaining elements
    for (unsigned int i = vec_n * 4 + idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sub_f32(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int vec_n = n / 4;
    for (unsigned int i = idx; i < vec_n; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x - vb.x;
        vc.y = va.y - vb.y;
        vc.z = va.z - vb.z;
        vc.w = va.w - vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }
    for (unsigned int i = vec_n * 4 + idx; i < n; i += stride) {
        c[i] = a[i] - b[i];
    }
}

__global__ void mul_f32(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int vec_n = n / 4;
    for (unsigned int i = idx; i < vec_n; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x * vb.x;
        vc.y = va.y * vb.y;
        vc.z = va.z * vb.z;
        vc.w = va.w * vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }
    for (unsigned int i = vec_n * 4 + idx; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }
}

__global__ void div_f32(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = a[i] / b[i];
    }
}

// ============================================================================
// Scalar ops: C[i] = op(A[i], scalar)
// ============================================================================

__global__ void add_scalar_f32(const float* a, float scalar, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = a[i] + scalar;
    }
}

__global__ void mul_scalar_f32(const float* a, float scalar, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = a[i] * scalar;
    }
}

// ============================================================================
// Unary ops: C[i] = op(A[i])
// ============================================================================

__global__ void neg_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = -a[i];
    }
}

__global__ void abs_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = fabsf(a[i]);
    }
}

__global__ void sqrt_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = sqrtf(a[i]);
    }
}

__global__ void exp_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = expf(a[i]);
    }
}

__global__ void log_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = logf(a[i]);
    }
}

// ============================================================================
// Activation functions
// ============================================================================

__global__ void relu_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int vec_n = n / 4;
    for (unsigned int i = idx; i < vec_n; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vc;
        vc.x = fmaxf(va.x, 0.0f);
        vc.y = fmaxf(va.y, 0.0f);
        vc.z = fmaxf(va.z, 0.0f);
        vc.w = fmaxf(va.w, 0.0f);
        reinterpret_cast<float4*>(c)[i] = vc;
    }
    for (unsigned int i = vec_n * 4 + idx; i < n; i += stride) {
        c[i] = fmaxf(a[i], 0.0f);
    }
}

__global__ void gelu_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    for (unsigned int i = idx; i < n; i += stride) {
        float x = a[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        c[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void silu_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    for (unsigned int i = idx; i < n; i += stride) {
        float x = a[i];
        c[i] = x / (1.0f + expf(-x));
    }
}

__global__ void sigmoid_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = 1.0f / (1.0f + expf(-a[i]));
    }
}

__global__ void tanh_f32(const float* a, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = tanhf(a[i]);
    }
}

// ============================================================================
// Clamp
// ============================================================================

__global__ void clamp_f32(const float* a, float lo, float hi, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = fminf(fmaxf(a[i], lo), hi);
    }
}

// ============================================================================
// Pow
// ============================================================================

__global__ void pow_scalar_f32(const float* a, float exponent, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        c[i] = powf(a[i], exponent);
    }
}

} // extern "C"
