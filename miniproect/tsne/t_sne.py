import numpy as np
from sklearn.manifold import TSNE

tsne = TSNE()
X = np.loadtxt('fashion20k_matrixs.txt')
X_embeded = TSNE(n_components=2).fit_transform(X)
print(X_embeded.shape)

file_tsne = 'fashion20k_tsne.txt'
with open(file_tsne, 'wb') as h:
    np.savetxt(h, X_embeded)
print(" fashion20k_tsne.txt" + " has already saved! ")
