#!/usr/bin/python3
from filegroup import *


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
    print(n_atoms)
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
        print(coords_batch)

        image_batch = gen_img_torch_batch(
            coords_batch, grid, sigma, norm
        )  # generate a batch of images with ctfs=None

        images_cpu[start:end] = image_batch.cpu()  # send the batch of images to the cpu

    return images_cpu
