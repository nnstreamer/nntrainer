__kernel void sgemv_cl_noTrans(const __global float *A, const __global float *X,
                               __global float *Y, unsigned int N,
                               unsigned int lda) {
  unsigned int i;
  i = get_global_id(0);
  float y0 = 0.0f;
  for (unsigned int j = 0; j < N; j++)
    y0 += A[j + i * lda] * X[j];
  Y[i] = y0;
}
