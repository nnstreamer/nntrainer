#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TS 16
__kernel void sgemm_cl_noTrans_fp16(__global const half *A,
                                    __global const half *B, __global half *C,
                                    const int M, const int N, const int K) {
  const int globalRow = get_global_id(1); // M dimension
  const int globalCol = get_global_id(0); // N dimension

  const int localRow = get_local_id(1);
  const int localCol = get_local_id(0);
  const int groupRow = TS * get_group_id(1);
  const int groupCol = TS * get_group_id(0);

  __local half Asub[TS][TS];
  __local half Bsub[TS][TS];

  float sum = 0.0f;

  for (int t = 0; t < (K + TS - 1) / TS; ++t) {
    const int tiledRowA = groupRow + localRow;
    const int tiledColA = t * TS + localCol;

    const int tiledRowB = t * TS + localRow;
    const int tiledColB = groupCol + localCol;

    // Load A
    if (tiledRowA < M && tiledColA < K)
      Asub[localRow][localCol] = A[tiledRowA * K + tiledColA];
    else
      Asub[localRow][localCol] = (half)0.0h;

    // Load B
    if (tiledRowB < K && tiledColB < N)
      Bsub[localRow][localCol] = B[tiledRowB * N + tiledColB];
    else
      Bsub[localRow][localCol] = (half)0.0h;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TS; ++k)
      sum += (float)(Asub[localRow][k]) * (float)(Bsub[k][localCol]);

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N)
    C[globalRow * N + globalCol] = (half)(sum);
}
