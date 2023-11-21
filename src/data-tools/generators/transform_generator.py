#!/usr/bin/python3
from filegroup import *


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


def quaternion_to_matrix(quaternions):
    """
    quaternion_to_matrix : function : Convert rotations given as quaternions to rotation matrices

    Input:
        quaternions: tensor of float shape (4) : quaternions leading with the real part
    Output:
        rot_mat : tensor of shape (3, 3) : Rotation matrices
    """
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
    """
    gen_quat_torch : function : sample quaternions from spherically uniform random distribution of directions

    Input:
        num_quaternions: int : number of quaternions generated
    Output:
        quat_out : tensor of shape (num_quaternions, 4) : quaternions generated
    """
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


def apply_ctf_batch(image, ctfs):
    """
    apply ctf to image
    """
    ft_image = torch.fft.fft2(image, dim=(1, 2), norm="ortho")  # ft of image
    image_ctf = torch.real(
        torch.fft.ifft2(ctfs * ft_image, dim=(1, 2), norm="ortho")
    )  # real of inverse fft?
    return image_ctf  # return image with ctf applied


def circular_mask(n_pixel, radius=0.4):
    """
    circular_mask : function : define a circular mask centered at center of the image for SNR calculation purpose (see Method for detail)

    Input :
        n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
        radius : float : radius of the circular mask relative to n_pixel, when radius = 0.5, the circular touches the edges of the image
    Output :
        mask : torch tensor of bool of shape (N_pixel, N_pixel) : circular mask to be applied onto the image
    """
    grid = torch.linspace(
        -0.5 * (n_pixel - 1), 0.5 * (n_pixel - 1), n_pixel
    )  # create a square to fit circle
    grid_x, grid_y = torch.meshgrid(
        grid, grid, indexing="ij"
    )  # create a meshgrid from the grid
    r_2d = grid_x**2 + grid_y**2  # region of pixels in the mask
    mask = r_2d < radius**2  # boundary of the mask
    return mask  # return the mask


def add_noise_torch_batch(img, snr, device="cuda"):
    """
    add_noise_torch_batch : function : add colorless Gaussian pixel noise to images

    Input :
        n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
        snr : float : Signal-to-noise (SNR) for adding noise to the image, if snr = np.infty, does not add noise to the images
    Output :
        image_noise : torch tensor of float of shape (N_image, N_pixel, N_pixel) : synthetic images with added noise
    """
    n_pixel = img.shape[1]  # number of pixels in the image
    radius = n_pixel * 0.4  # radius of each gaussian pixel
    mask = circular_mask(n_pixel, radius)  # create a circular mask
    image_noise = torch.empty_like(
        img, device=device
    )  # initialize tensor for images with noise
    for i, image in enumerate(img):  # account for each image
        image_masked = image[mask]  # mask the image
        signal_std = (
            image_masked.pow(2).mean().sqrt()
        )  # calculate standard deviation of the image
        noise_std = signal_std / np.sqrt(
            snr
        )  # calculate noise by deviding std by sqrt of snr
        noise = torch.distributions.normal.Normal(0, noise_std).sample(
            image.shape
        )  # sample normal distribution with std for noise
        image_noise[i] = image + noise  # add noise to the image
    return image_noise  # return tensor containing images with noise added
