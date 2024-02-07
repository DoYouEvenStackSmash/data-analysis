#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *


def original_likelihood(omega, m, N_pix, noise=1):
    """
    Likelihood function from the paper
    L(\omega, m, N_{\text{pix}}, \text{noise}) = \frac{1}{{(2 \cdot \text{noise}^2 \cdot \pi)^{\frac{N_{\text{pix}}}{2}}}} \cdot \exp\left(-\frac{{\|\omega - m\|^2}}{{2 \cdot \text{noise}^2}}\right)

    """
    L = np.power(
        2.0 * np.square(noise) * np.pi, -1.0 * (np.divide(N_pix, 2.0))
    ) * np.exp(
        -1.0
        * np.divide(np.square(np.linalg.norm(omega - m)), (2.0 * (np.square(noise))))
    )
    return L


def postprocessing_adjust(input_arr, noise, const_factor=0):
    """
    Placeholder for modifying an array
    -\log \left(
        \sqrt{2\pi \lambda^2}
    \right)
    """
    log_add = 0
    const_factor = 1

    # branching for easy enable/disable
    if True:
        log_add = np.log(const_factor * np.sqrt(2 * np.pi * noise**2))

    return [np.log(i) - log_add for i in input_arr]


def likelihood(omega, m, N_pix, noise=1):
    """
    New likelihood
    \log \left(
        \sum_{j=1}^M \weight_j
        \exp \left( - \left\| \omega_i - x_j \right\|^2 / 2 \lambda^2 \right)
    \right)
    """
    lambda_square = noise**2
    coeff = 1
    l2 = jnp.exp(-1.0 * (jnp.square(custom_distance(omega, m)) / (2 * lambda_square)))
    return coeff * l2
