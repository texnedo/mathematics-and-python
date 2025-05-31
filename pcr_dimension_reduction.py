import numpy as np
from sklearn.decomposition import PCA


def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    ### START CODE HERE ###
    # mean center the data
    mean = np.mean(X, axis=0, keepdims=True)
    X_demeaned = (X - mean)

    # print("Data:\n", X)
    # print("Normalized Data:\n", X_demeaned)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    # print("Covariance:\n", covariance_matrix)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    # print("Eigen Vals:\n", eigen_vals)
    # print("Eigen Vecs:\n", eigen_vecs)

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)

    # print("Sorted Indexes:\n", idx_sorted)

    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # print("Reversed Indexes:\n", idx_sorted_decreasing)

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

    # print("Sorted Eigen Vecs:\n", eigen_vecs_sorted)

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or n_components)
    eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]

    # print("Top K Eigen Vecs:\n", eigen_vecs_subset)

    print(eigen_vecs_subset.shape)
    print(X_demeaned.shape)

    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product. Note that, since for any matrices A, B, (A.B).T = B.T . A.T,
    # this reduces to the dot product of the de-mean data with the eigenvectors
    X_reduced = np.dot(X_demeaned, eigen_vecs_subset)

    ### END CODE HERE ###

    return X_reduced


def main():
    # Testing your function
    np.random.seed(1)
    X = np.random.rand(3, 10)
    X_reduced = compute_pca(X, n_components=2)
    print("Your original matrix was " + str(X.shape) + " and it became:")
    print(X_reduced)

    pca = PCA(n_components=2)  # Reduce to 2 components (for example)
    transformed_data = pca.fit_transform(X)

    print("\nTransformed Data (PCA):\n", transformed_data)


if __name__ == '__main__':
    main()