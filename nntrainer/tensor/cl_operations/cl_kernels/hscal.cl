#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sscal_cl_fp16(__global half *X, const float alpha) {

  unsigned int i = get_global_id(0);
  X[i] *= alpha;
}
