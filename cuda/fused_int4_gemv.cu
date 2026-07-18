// fused_int4_gemv.cu
//
// Fused (dequantize INT4 weight group) + (GEMV against INT8 activations) in
// a single kernel launch, avoiding a materialized fp16 weight tensor.
//
// Grid:  one block per output row (out_features blocks)
// Block: `threads` lanes cooperatively reduce over in_features
//
// Weight layout (must match engine/quantization.py WeightQuantizer):
//   qw      : uint8[out_features][in_features/2]   two INT4 values per byte
//             (low nibble = even column, high nibble = odd column)
//   scales  : float32[out_features][n_groups]
//   zeros   : float32[out_features][n_groups]       zero-point, in int-space
//   x       : int8[in_features]                     quantized activations
//   x_scale : float32                                dequant scale for x
//
// y[row] = sum_col dequant(qw[row][col]) * (x[col] * x_scale)

extern "C" __global__
void fused_int4_gemv(
    const unsigned char* __restrict__ qw,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const signed char* __restrict__ x,
    float x_scale,
    float* __restrict__ y,
    int out_features,
    int in_features,
    int group_size,
    int n_groups)
{
    int row = blockIdx.x;
    if (row >= out_features) return;

    extern __shared__ float partial[];
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    const unsigned char* qw_row = qw + (size_t)row * (in_features / 2);
    const float* scale_row = scales + (size_t)row * n_groups;
    const float* zero_row = zeros + (size_t)row * n_groups;

    float acc = 0.0f;
    // Each thread strides over packed bytes (2 weights per byte).
    int packed_len = in_features / 2;
    for (int b = tid; b < packed_len; b += n_threads) {
        unsigned char byte = qw_row[b];
        int col0 = 2 * b;
        int col1 = col0 + 1;

        int g0 = col0 / group_size;
        int g1 = col1 / group_size;

        float w0 = ((float)(byte & 0x0F) - zero_row[g0]) * scale_row[g0];
        float w1 = ((float)((byte >> 4) & 0x0F) - zero_row[g1]) * scale_row[g1];

        float xv0 = (float)x[col0] * x_scale;
        float xv1 = (float)x[col1] * x_scale;

        acc += w0 * xv0 + w1 * xv1;
    }

    partial[tid] = acc;
    __syncthreads();

    // Tree reduction within the block.
    for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[row] = partial[0];
    }
}
