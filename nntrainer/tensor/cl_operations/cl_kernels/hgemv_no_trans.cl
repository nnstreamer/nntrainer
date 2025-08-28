#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sgemv_cl_noTrans_fp16(const __global half *A,
                                    const __global half *X, __global half *Y,
                                    unsigned int N, unsigned int lda) {
  unsigned int i;
  i = get_global_id(0);
  float y0 = 0.0f;
  for (unsigned int j = 0; j < N; j++)
    y0 += A[j + i * lda] * X[j];
  Y[i] = y0;
}
