#!/usr/bin/python3
from filegroup import *


def generate_ctfs(N_images, hyperparams=None, seed=12345):
    """Generates some number of ctfs

    Args:
        N_images (_type_): _description_
        hyperparams (_type_, optional): _description_. Defaults to None.
        seed (int, optional): _description_. Defaults to 12345.

    Returns:
        _type_: _description_
    """
    amp = 0.1  ## Amplitude constrast ratio
    b_factor = 1.0  ## B-factor
    defocus_min, defocus_max = [0.5, 4.0]
    n_pixel = 128
    pixel_size = 0.3

    if hyperparams != None:
        amp = hyperparams.amplitude[0]
        b_factor = hyperparams.bFactor[0]
        seed = hyperparams.seed
        defocus = torch.tensor(hyperparams.defocus, dtype=torch.float64)
        N_images = defocus.shape[0]
    else:
        defocus = (
            torch.rand(N_images, dtype=torch.float64)  # set defocus value
            * (defocus_max - defocus_min)
            + defocus_min
        )  ## defocus

    elecwavel = 0.019866  ## electron wavelength in Angstrom
    gamma = defocus * (
        np.pi * 2.0 * 10000 * elecwavel
    )  ## gamma coefficient in SI equation 4 that include the defocus
    # print(gamma.shape)
    freq_pix_1d = torch.fft.fftfreq(
        n_pixel, d=pixel_size, dtype=torch.float64
    )  # get the fft bins
    freq_x, freq_y = torch.meshgrid(
        freq_pix_1d, freq_pix_1d, indexing="ij"
    )  # create a 2d grid for x and y
    freq2_2d = freq_x**2 + freq_y**2  ## square of modulus of spatial frequency

    ctf_batch = _calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor)
    return ctf_batch


def _calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor):
    """
    calc_ctf_torch_batch : function : generate random Contrast transfer function (CTF)
    called by generate_ctfs

    Input :
            freq2_2d : torch tensor of float of shape (N_pixel, N_pixel) : square of modulus of spatial frequency in Fourier space
            amp : float : Amplitude constrast ratio
            gamma : torch tensor of float of shape (N_image) : gamma coefficient in SI equation 4 that include the defocus
            b_factor : float : B-factor
    Output :
            ctf : torch tensor of float of shape (N_image, N_pixel, N_pixel) : randomly generated CTF
    """
    # env = torch.exp(- b_factor.view(-1,1,1) * freq2_2d.unsqueeze(0) * 0.5)
    # ctf = amp.view(-1,1,1) * torch.cos(gamma.view(-1,1,1) * freq2_2d * 0.5) - torch.sqrt(1 - amp.view(-1,1,1) **2) * torch.sin(gamma.view(-1,1,1)  * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
    env = torch.exp(-b_factor * freq2_2d.unsqueeze(0) * 0.5)
    ctf = (
        amp * torch.cos(gamma.view(-1, 1, 1) * freq2_2d * 0.5)
        - np.sqrt(1 - amp**2) * torch.sin(gamma.view(-1, 1, 1) * freq2_2d * 0.5)
        + torch.zeros_like(freq2_2d) * 1j
    )
    ctf *= env
    return ctf
