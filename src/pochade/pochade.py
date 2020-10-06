from matplotlib.image import imread
import numpy as np
# from skimage import color as skic
# from .colors import xyz2lab, rgb2xyz, rgb2lab
import random


def get_rgb(number):
    """
    Extract 8bit ints from 24bit str

    :param number:
    :return: [r, g, b] value from extracted 8bit integer
    """
    bit_str = np.binary_repr(number, width=24)
    lab = []
    for i in range(0, 24, 8):
        lab.append(int(bit_str[i:i + 8], 2))

    return lab


def sample(img):
    mask = 0b111000  # mask for 3 most significant bits
    c = Counter()
    for row in img:
        for rgb in row:
            lab = colors.rgb2lab(rgb)
            lab = np.rint(lab).astype(np.int16)
            # lab = rgb

            lab_bin = np.zeros(3).astype(np.int8)  # CIE-LAB might be -128 to 128
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


def gla_slow(img, K=6):
    """
    Perform K-means clustering on img data.

    Parameters
    ----------
    img : numpy array of RGB data
    K : int Number of clusters.


    Returns (as tuple)
    -------
    groups   : N x 1 array containing cluster numbers of data at indices in X.
    centers    : K x M array of cluster centers.
    """
    max_iter = 100
    tol = 1e-4

    # reshape image into a list of rgb tuples
    new_row = img.shape[0] * img.shape[1]
    img = np.reshape(img, (new_row, img.shape[2]))

    print(img.shape)
    n = img.shape[0]
    rand = np.random.permutation(n)  # random cluster initialization
    centers = img[rand[0:K], :]

    iter = 1
    groups = np.zeros((n,))  # store cluster assignments
    done = False
    dist_sum = np.inf
    old_dist_sum = 0

    while not done:
        error = 0
        for i in range(n):
            # compute distances for each cluster center
            dists = np.sum((centers - img[i, :]) ** 2, axis=1)
            # print(img[i, :].shape)
            # print(dists)

            idx = np.argmin(dists, axis=0)  # find index closest cluster
            # print(idx)
            groups[i] = idx                 # assign i to cluster
            min_dist = dists[idx]
            error += min_dist               # running sum to error to closest cluster

        for j in range(K):  # now update each cluster center j...
            if np.any(groups == j):
                centers[j, :] = np.mean(img[(groups == j).flatten(), :], 0)  # mean of the assigned data
            else:
                centers[j, :] = img[int(np.floor(np.random.rand())), :]  # random restart if no assigned data

        iter += 1
        done = (iter > max_iter) or (error == old_dist_sum)
        old_dist_sum = error

    return groups, centers


def closest_centroid(img, centroids):
    """
    Returns indices from each point to the closest centroid using the norm.

    :param img: Reshaped np.array for the pixels in the image.
    :param centroids: np.array of the coordinates for each centroid
    :return: np.array containing the index of the cluster that each pixel is closest too
    """
    dist = np.linalg.norm(img[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    closest_centroid_index = np.argmin(dist, axis=1)

    return closest_centroid_index


def gla_fast(img, nclusters=6):
    """
    Run generalized Lloyd's algorithm to cluster colors.

    :param img: Reshaped np.array for the pixels in the image
    :param nclusters: The esired number of clusters
    :return: Centroids representing the mean color for each cluster
    """
    rand_idx = random.sample(range(img.shape[0]), nclusters)
    centroids = img[rand_idx]                       # randomly init clusters
    # print(len(centroids))
    assignments = np.zeros(len(img), dtype=np.int32)   # prepare space for cluster assignments

    iters = 0
    max_iters = 300
    done = False

    while not done:
        assignments = closest_centroid(img, centroids)

        for c in range(nclusters):
            members = img[assignments == c]
            centroids[c] = members.mean(axis=0)

        done = (iters == max_iters)     # TODO: update done condition for early stopping
                                        #  by checking convergence against tol
        iters += 1

    return centroids

def palette(path, nccolors=6):
    """
    Calculate a color palette given a path to an image.

    :param path: String for the file path
    :param nccolors: Number of Desired colors

    :return:A (ncolors x 3) np.array where each row is an rgb tuple
    """
    img = imread(path)

    new_row = img.shape[0] * img.shape[1]
    img = np.reshape(img, (new_row, img.shape[2]))

    return gla_fast(img, ncolors)

""""
def main():
    from skimage import data
    from sklearn.cluster import KMeans
    import colorthief
    import time

    path = "/Users/laguna/Desktop/index.jpg"

    img = imread(path)
    a = time.time()

    # img = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # kmeans = KMeans(n_clusters=6).fit(img)
    palette = gla_fast(img)
    b = time.time()
    print(palette)
    print(b-a)

    # print(kmeans.cluster_centers_)
    # before optimization 16.97 seconds
main()
"""
