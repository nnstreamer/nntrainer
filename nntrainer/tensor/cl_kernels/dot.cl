__kernel void dot_cl(const __global float *A, const __global float *X,
                     unsigned int K, __global float *res) {
  *res = 0;
  for (unsigned int i = 0; i < K; i++) {
    *res += A[i] * X[i];
  }
}
