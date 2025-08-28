#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void dot_cl_fp16(const __global half *A, const __global half *X,
                          unsigned int K, __global half *res) {
  float y = 0.0f;
  for (unsigned int i = 0; i < K; i++) {
    y += A[i] * X[i];
  }
  *res = y;
}
