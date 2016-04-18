import numpy as np
import pandas
import math
import sklearn
import sklearn.cross_validation
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.svm
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.feature_extraction
import sklearn.decomposition
import scipy.sparse
import sklearn.ensemble
import matplotlib.pyplot as plt
import sklearn.cluster
import skimage.io


def get_new_image(shape, center_coords, centers):
    image = np.zeros(shape)
    cur = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i][j] = center_coords[centers[cur]]
            cur += 1
    return image


def psnr(image, new_image):
    mse = np.mean((image - new_image) ** 2)
    return 10. * np.log10((np.max(image) ** 2) / mse)


def save_image(image, clusters, psnr):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.savefig("%d_clusters_%.2f_psnr.png" % (clusters, psnr))
    plt.close()


if __name__ == '__main__':
    image = skimage.img_as_float(skimage.io.imread('parrots.jpg'))

    X = []
    for i in image:
        for j in i:
            X.append(j)
    X = np.array(X)

    n_clusters = 2
    while True:
        clst = sklearn.cluster.KMeans(init='k-means++', random_state=241, n_clusters=n_clusters)
        centers = clst.fit_predict(X)
        new_image = get_new_image(image.shape, clst.cluster_centers_, centers)
        psnr_ = psnr(image, new_image)

        save_image(new_image, n_clusters, psnr_)

        if psnr_ > 20:
            print n_clusters
            break
        n_clusters += 1
