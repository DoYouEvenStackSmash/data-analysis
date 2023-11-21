import torch
import numpy as np
from tqdm import tqdm


def gen_grid(n_pixel, pixel_size):
    """
    gen_grid : function : generate square grids of positions of each pixel

    Input:
        n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
        pixel_size : float : width of each pixel in physical space in Angstrom
    Output:
        grid : torch tensor of float of shape (N_pixel) : physical location of center of each pixel (in Angstrom)
    """
    grid_min = (
        -pixel_size * (n_pixel - 1) * 0.5
    )  # minimum position on the grid, below zero
    grid_max = (
        -grid_min
    )  # pixel_size*(n_pixel-1)*0.5    # maximum position on the grid, above zero(contains zero!)
    grid = torch.linspace(
        grid_min, grid_max, n_pixel
    )  # create a linear subspace with the bounds and division
    return grid


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
        amp = hyperparams["amp"]
        b_factor = hyperparams["b_factor"]
        seed = hyperparams["seed"]
        defocus_min = hyperparams["defocus_min"]
        defocus_max = hyperparams["defocus_max"]

    defocus = (
        torch.rand(N_images, dtype=torch.float64)  # set defocus value
        * (defocus_max - defocus_min)
        + defocus_min
    )  ## defocus

    elecwavel = 0.019866  ## electron wavelength in Angstrom
    gamma = defocus * (
        np.pi * 2.0 * 10000 * elecwavel
    )  ## gamma coefficient in SI equation 4 that include the defocus

    freq_pix_1d = torch.fft.fftfreq(
        n_pixel, d=pixel_size, dtype=torch.float64
    )  # get the fft bins
    freq_x, freq_y = torch.meshgrid(
        freq_pix_1d, freq_pix_1d, indexing="ij"
    )  # create a 2d grid for x and y
    freq2_2d = freq_x**2 + freq_y**2  ## square of modulus of spatial frequency

    ctf_batch = calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor)
    return ctf_batch


def gen_img_torch_batch(coord, grid, sigma=1.0, norm=None):
    """
    gen_img_torch_batch : function : generate images from atomic coordinates
    Input :
        coord : numpy ndarray or torch tensor of float of shape (N_image, N_atom, 3) : 3D Cartesian coordinates of atoms of configuration aligned to generate the synthetic images
        grid : torch tensor of float of shape (N_pixel) : physical location of center of each pixel (in Angstrom)
        sigma : float : Gaussian width of each atom in the imaging model in Angstrom
        norm : float : normalization factor for image intensity
        ctfs : torch tensor of float of shape (N_image, N_pixel, N_pixel) : random generated CTF added to each of the synthetic image
    Output :
        image or image_ctf : torch tensor of float of shape (N_image, N_pixel, N_pixel) : synthetic images with or without randomly generated CTF applied
    """
    gauss_x = -0.5 * ((grid[:, :, None] - coord[:, :, 0]) / sigma) ** 2  ##
    gauss_y = (
        -0.5 * ((grid[:, :, None] - coord[:, :, 1]) / sigma) ** 2
    )  ## pixels are square, grid is same for x and y directions
    gauss = torch.exp(gauss_x.unsqueeze(1) + gauss_y)
    image = gauss.sum(3) * norm  # normalize the image
    image = image.permute(2, 0, 1)  # wrap..the axis?

    return image  #


def apply_ctf_batch(image, ctfs):
    """
    apply ctf to image
    """
    ft_image = torch.fft.fft2(image, dim=(1, 2), norm="ortho")  # ft of image
    image_ctf = torch.real(
        torch.fft.ifft2(ctfs * ft_image, dim=(1, 2), norm="ortho")
    )  # real of inverse fft?
    return image_ctf  # return image with ctf applied


def calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor):
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


def quaternion_to_matrix(quaternions):
    ##
    ## quaternion_to_matrix : function : Convert rotations given as quaternions to rotation matrices
    ##
    ## Input:
    ##     quaternions: tensor of float shape (4) : quaternions leading with the real part
    ## Output:
    ##     rot_mat : tensor of shape (3, 3) : Rotation matrices
    ##
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    rot_mat = o.reshape(quaternions.shape[:-1] + (3, 3))
    return rot_mat


def gen_quat_torch(num_quaternions, device="cuda"):
    ##
    ## gen_quat_torch : function : sample quaternions from spherically uniform random distribution of directions
    ##
    ## Input:
    ##     num_quaternions: int : number of quaternions generated
    ## Output:
    ##     quat_out : tensor of shape (num_quaternions, 4) : quaternions generated
    ##
    over_produce = 5  ## for ease of parallelizing the calculation, it first produce much more than the needed amount of quanternion, then filter the ones that satisfy the condition
    quat = (
        torch.rand((num_quaternions * over_produce, 4), dtype=torch.float64) * 2.0 - 1.0
    )  # select a number of quaternions
    norm = torch.linalg.vector_norm(
        quat, ord=2, dim=1
    )  # calculate norm of quaternions to make sure none of them are too close?
    quat /= norm.unsqueeze(1)  # divide quaternions by unsqueezed norm
    good_ones = torch.bitwise_and(
        torch.gt(norm, 0.2), torch.lt(norm, 1.0)
    )  ## this condition, norm of quaternion has to be < 1.0 and > 0.2, has to be satisfied
    quat_out = quat[good_ones][:num_quaternions]  ## just chop the ones needed
    return quat_out


def generate_rotations(coord, num_rotations, rotation=True):
    """Generates rotations and applies them to coord

    Args:
        coord (_type_): _description_
        num_rotations (_type_): _description_
        rotation (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if rotation:  # compute rotation matrices and... coordinate transforms
        # get_rotations()
        quats = gen_quat_torch(num_rotations)  # sample quaternions
        rot_mats = quaternion_to_matrix(quats).type(
            torch.float64
        )  # convert quaterions to 3x3 rotation matrices
        # rot_mats = rot_mats.to(device)  # send rotation matrices to device
        coord_rot = coord.matmul(
            rot_mats
        )  # compute coordinate rotation matrices by multiplying coordinates by rotation matrix
    else:
        rot_mats = (
            torch.eye(3).unsqueeze(0).repeat(num_rotations, 1, 1).type(torch.float64)
        )  # create identity matrices?
        coord_rot = coord  # coordinates rotated are just coordinates
    return coord_rot


def simulate_images(coord, grid, sigma, batch_size=32):
    """
    simulate_images : function : generate synthetic cryo-EM images, at random orientation (and random CTF), given a set of structures

    Input :
        coord : numpy ndarray or torch tensor of float of shape (N_image, N_atom, 3) : 3D Cartesian coordinates of atoms of configuration aligned to generate the synthetic images
        n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
        pixel_size : float : width of each pixel in physical space in Angstrom
        sigma : float : Gaussian width of each atom in the imaging model in Angstrom
        snr : float : Signal-to-noise (SNR) for adding noise to the image, if snr = np.infty, does not add noise to the images
        add_ctf : bool : If True, add Contrast transfer function (CTF) to the synthetic images.
        batch_size : int : to split the set of images into batches for calculation, where structure in the same batch are fed into calculation at the same time, a parameter for computational performance / memory management
        device : str : "cuda" or "cpu", to be fed into pyTorch, see pyTorch manual for more detail
    Output :
        rot_mats : torch tensor of float of shape (N_image, 3, 3) : Rotational matrices randomly generated to orient the configraution during the image generation process
        ctfs_cpu : torch tensor of float of shape (N_image, N_pixel, N_pixel) : random generated CTF added to each of the synthetic image
        images_cpu : torch tensor of float of shape (N_image, N_pixel, N_pixel) : generated synthetic images
    """

    if type(coord) == np.ndarray:  # convert numpy arrays to torch tensors
        coord = torch.from_numpy(coord).type(torch.float64)
    # coord = coord.to(device)  # send coordinate tensor to device

    n_pixel = grid.shape[0]
    # how many CTFS to generate
    n_struc = coord.shape[0]  # get total number of structures
    n_atoms = coord.shape[1]  # get total number of atoms
    norm = 0.5 / (np.pi * sigma**2 * n_atoms)  # likelihood normalization constant

    N_images = n_struc  # number of images is the sae as number of structures

    n_batch = int(N_images / batch_size)  # compute number of batches

    if (
        n_batch * batch_size < N_images
    ):  # correct off by 1 to make sure that the last batch is fileld
        n_batch += 1

    images_cpu = torch.empty(
        (N_images, n_pixel, n_pixel),  # create a tensor for synthesized images
        dtype=torch.float64,
        # device="cpu",
    )

    for i in tqdm(
        range(n_batch), desc="Generating images for batch"
    ):  # generate batch of images
        start = i * batch_size  # choose a start index
        end = (i + 1) * batch_size  # choose an end index
        coords_batch = coord[start:end]  # get a batch of transformed coordinates

        image_batch = gen_img_torch_batch(
            coords_batch, grid, sigma, norm
        )  # generate a batch of images with ctfs=None

        images_cpu[start:end] = image_batch.cpu()  # send the batch of images to the cpu

    return images_cpu
