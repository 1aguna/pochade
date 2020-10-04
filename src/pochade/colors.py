import numpy as np


def rgb2xyz(rgb):
    """
    Converts sRGB image to CIE XYZ. Assumes D65 reference white

    .. [1] Transformation matrix from http://www.brucelindbloom.com
    .. [2] Gamma correction from http://wiki.nuaj.net/index.php/Color_Transforms#RGB_.E2.86.92_XYZ

    :param rgb: Color in RGB Mode
    :return: Image in CIE XYZ space
    """

    # Transformation matrix from sRGB to CIE XYZ
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])

    # Convert rgb image from int [0-255] to float [0-1]
    arr = rgb / 255.0

    # Gamma correction
    arr = np.where(arr > 0.0405,
                   ((arr + 0.055) / 1.055) ** 2.4,
                   arr / 12.92)

    return np.matmul(arr, m.T)



def xyz2lab(xyz):
    """
    Converts image in XYZ color-space to CIE Lab
    .. [1] scikit-image color module: https://github.com/scikit-image/scikit-image

    :param xyz: Image in XYZ color space
    :return: Image in CIE-LAB color space

    """
    xyz_ref_white = np.array([.95047, 1.00, 1.08883])

    arr = xyz.copy().astype(float)
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    arr = np.where(arr > 0.008856,
                   np.cbrt(arr),
                   (7.787 * arr) + (16 / 116.))

    x, y, z = arr

    # Vector scaling
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return np.array([L, a, b])


def rgb2lab(input):
    xyz = rgb2xyz(input)
    return xyz2lab(xyz)
