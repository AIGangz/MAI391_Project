from sklearn.decomposition import PCA
import numpy as np

np.random.seed(1779)



X = np.array([[0,7,5],[1,4,6],[7,2,4],[3,2,4],[6,8,3]])
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
#print(X_std)

pca = PCA(n_components=3)

X_std_tranformed = pca.fit_transform(X_std)

print(pca.components_)
print(pca.explained_variance_ratio_)
print(X_std_tranformed)
