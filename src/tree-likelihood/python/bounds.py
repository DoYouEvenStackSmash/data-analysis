#!/usr/bin/python3

def check_ctf_bound(mulk, yi,noise=1,tau=1e-6):
  pass
  R = centroid.cluster_radius
  -2 * noise*log(tau)
  if difference(mulk.m1,yi.m1) - mulk.cluster_radius > 0:
    
  #(||yi - center|| - bound) ** 2 >=  -2 * noise*log(tau)
  Cm = jax_apply_d1m2_to_d2m1(centroid.val, centroid.val)
  max_filter_yi = jax_apply_d1m2_to_d2m1(centroid.val, T)
  # ||F(yi) - F(omega)|| >= ||F(yi) - Cphi*F(mulk)|| - max||Cmax*F(omega_j_in_cluster) - Cmax*F(mulk)||
  # || yi - xi|| >= ||yi - Cp * mulk|| - max_over_j||Cmax*F(omega_j_in_cluster) - Cmax*mulk||
  # || yi - xi|| >= difference(T.m1, jax_apply_d1m2_to_d2m1(T,mulk.val)) - mulk.cluster_radius