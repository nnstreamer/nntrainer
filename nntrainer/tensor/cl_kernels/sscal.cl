__kernel void sscal_cl(__global float *X, const float alpha) {

  unsigned int i = get_global_id(0);
  X[i] *= alpha;
}
