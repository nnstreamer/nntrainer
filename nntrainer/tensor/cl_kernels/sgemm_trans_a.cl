#define TS 16
__kernel void sgemm_cl_transA(__global const float *A, __global const float *B,
                              __global float *C, const int M, const int N,
                              const int K) {
  const int globalRow = get_global_id(1); // M
  const int globalCol = get_global_id(0); // N

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  float sum = 0.0f;

  const int localRow = get_local_id(1);
  const int localCol = get_local_id(0);
  const int groupRow = TS * get_group_id(1);
  const int groupCol = TS * get_group_id(0);

  for (int t = 0; t < (K + TS - 1) / TS; ++t) {
    const int tiledRowA = t * TS + localCol;
    const int tiledColA = groupRow + localRow;

    if (tiledRowA < K && tiledColA < M)
      Asub[localRow][localCol] = A[tiledRowA * M + tiledColA];
    else
      Asub[localRow][localCol] = 0.0f;

    const int tiledRowB = t * TS + localRow;
    const int tiledColB = groupCol + localCol;

    if (tiledRowB < K && tiledColB < N)
      Bsub[localRow][localCol] = B[tiledRowB * N + tiledColB];
    else
      Bsub[localRow][localCol] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TS; ++k)
      sum += Asub[localRow][k] * Bsub[k][localCol];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[globalRow * N + globalCol] = sum;
}
