import cv2
import numpy as np


def train(img_vec, eig_vec_num):
    mean = np.mean(img_vec, 1)

    # Calculate covariance matrix
    pca_data = np.transpose(np.transpose(img_vec) - mean)
    pca_mat = np.dot(np.transpose(pca_data), pca_data)
    cov, temp_mean = cv2.calcCovarMatrix(pca_mat, mean,
                                         cv2.COVAR_NORMAL | cv2.COVAR_ROWS)

    # Eigenvectors and eigenvalues
    eig_val_temp, eig_vec_temp = cv2.eigen(cov, eigenvectors=True)[1:]
    eig_vec = np.dot(eig_vec_temp, np.transpose(img_vec))
    eig_vec = (eig_vec - eig_vec.min()) / (eig_vec.max() - eig_vec.min())

    eig_vec_mat = []
    for idx in range(eig_vec_num):
        eig_vec_mat.append(eig_vec[idx])

    return mean, eig_vec_mat
