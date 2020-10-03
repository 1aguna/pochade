from matplotlib.image import imread
import numpy as np
from skimage import color as skic
import colors
from collections import Counter
import colorgram
import random


def set_bits(num, value):
    """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index  # Compute mask, an integer with just bit 'index' set.
    v &= ~mask  # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask  # If x was True, set the bit indicated by the mask.
    return v  # Return the result, we're done.


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


def palette(img, ncolors=6, max_iters=100, method="kmeans"):
    # img = np.array(Image.open(fpath))

    counts = sample(img)
    for color, _ in counts.most_common(ncolors):
        print(get_rgb(color))


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

    dist = np.linalg.norm(img[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    closest_centroid_index = np.argmin(dist)

    return closest_centroid_index


def gla_fast(img, nclusters=6):
    rand_idx = random.sample(range(img.shape[0]), nclusters)
    centroids = img[rand_idx]                       # randomly init clusters
    assignments = np.zeros(len(img), dtype=np.int32)   # prepare space for cluster assignments

    iters = 0
    max_iters = 300
    done = False

    while not done:
        assignments = closest_centroid(img, centroids)

        for c in range(nclusters):
            members = img[assignments == c]
            centroids[c, :] = members.mean(axis=0)

        done = (iters == max_iters)     # TODO: update done condition for early stopping
                                        #  by checking convergence against tol
        iters += 1
    return centroids



def main():
    from skimage import data
    from sklearn.cluster import KMeans
    import time

    path = "/Users/laguna/Desktop/index.jpg"

    img = imread(path)
    a = time.time()

    # img = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # kmeans = KMeans(n_clusters=6).fit(img)
    palette = gla(img)
    b = time.time()
    print(b-a)

    # print(kmeans.cluster_centers_)
    # before optimization 16.97 seconds
main()
