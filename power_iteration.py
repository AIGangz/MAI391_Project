import numpy as np

def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k

a = power_iteration(np.array([[0.5, 0.5], [0.2, 0.8]]), 100)
print(a)

print(np.linalg.eig(np.array([[0.5, 0.5], [0.2, 0.8]])))