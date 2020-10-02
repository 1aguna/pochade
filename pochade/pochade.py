from matplotlib.image import imread
import numpy as np
from skimage import color as skic
import colors
from collections import Counter
import colorgram

def set_bits(num, value):
    """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask         # If x was True, set the bit indicated by the mask.
    return v            # Return the result, we're done.


def get_rgb(number):
    """
    Extract 8bit ints from 24bit str

    :param number:
    :return: [r, g, b] value from extracted 8bit integer
    """
    bit_str = np.binary_repr(number, width=24)
    lab = []
    for i in range(0,24,8):
        lab.append(int(bit_str[i:i +8], 2))

    return lab


def palette(img, ncolors=6, max_iters=100, method="kmeans"):
    # img = np.array(Image.open(fpath))

    counts = sample(img)
    for color, _ in counts.most_common(ncolors):
        print(get_rgb(color))




def sample(img):
    mask = 0b111000    # mask for 3 most significant bits
    c = Counter()
    for row in img:
        for rgb in row:
            lab = colors.rgb2lab(rgb)
            lab = np.rint(lab).astype(np.int16)
            # lab = rgb

            lab_bin = np.zeros(3).astype(np.int8)   # CIE-LAB might be -128 to 128
            for i in range(3):
                sign = int(lab[i] < 0)
                sig_bits = lab[i] & mask
                lab_bin[i] = (sign << 8) | sig_bits
                # print(bin(sig_bits))

            # print(lab_bin, lab)
            # return 0

            binary_concat = "".join([np.binary_repr(x, width=8) for x in lab_bin])

            c[int(binary_concat, 2)] += 1

    return c



def kmeans(img, K=6, max_iter=100):
    """
    Perform K-means clustering on img data.

    Parameters
    ----------
    img : numpy array of RGB data
    K : int
        Number of clusters.
    max_iter : int (optional)
        Maximum number of optimization iterations.

    Returns (as tuple)
    -------
    z    : N x 1 array containing cluster numbers of data at indices in X.
    c    : K x M array of cluster centers.
    sumd : (scalar) sum of squared euclidean distances.
    """
    n = img.shape[0]
    rand = np.random.permutation(n)  # random cluster initialization
    centers = img[rand[0:K], :]

    # Now, optimize the objective using coordinate descent:
    iter = 1
    z = np.zeros((n,))

    while iter > max_iter:
        for i in range(n):
            # compute distances for each cluster center
            dists = np.sum((centers - img[i, :]) ** 2, axis=1)

        # print z
        for j in range(K):  # now update each cluster center j...
            if np.any(z == j):
                centers[j, :] = np.mean(img[(z == j).flatten(), :], 0)  # ...to be the mean of the assigned data...
            else:
                centers[j, :] = img[int(np.floor(np.random.rand())), :]  # ...or random restart if no assigned data
        iter += 1

    return centers


def main():
    from skimage import data
    import time
    from colorthief import ColorThief
    img = data.chelsea()

    img[0][0][0] = 51
    img[0][0][1] = 112
    img[0][0][2] = 235

    a = time.time()
    path = "/Users/laguna/Desktop/index.jpg"
    color_thief = ColorThief(path)
    palette = color_thief.get_palette(color_count=6)
    b = time.time()
    print(b-a)
    # img = imread(path)
    # a = time.time()
    # palette(img)
    # b = time.time()
    # print(b-a)

    # 2.92 for pochade

    # cs = colorgram.extract(path, 6)
    # for c in cs:
    #     print(c.rgb)

    # # test = img[0, 0]
    # test = np.array([255, 0, 0])
    # res = colors.rgb2xyz(test)
    # res = colors.xyz2lab(res)
    # rounded = np.rint(res).astype(np.int8)
    # print(rounded, rounded.dtype)


    mask = 0b110000000000  # mask for 2 most significant bits


main()
