import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from time import time


class CMeans:
    def __init__(self, C=3, m=2, max_iter=15, tolerance=1e-4):
        self.C = C
        self.m = m
        self.max_iter = max_iter
        self.tolerance = tolerance

    def get_centers(self):
        return self.__centers

    def get_partition(self):
        return self.__w

    def __update_centroids(self):
        self.__centers = np.stack(
            [
                np.divide(
                    np.sum(
                        np.multiply(
                            self.__data, np.power(self.__w[:, i], self.m)[..., None]
                        ),
                        axis=0,
                    ),
                    np.sum(np.power(self.__w[:, i], self.m)),
                )
                for i in range(self.C)
            ],
            axis=0,
        )

    def __update_w(self):
        self.__w = np.divide(
            1,
            np.stack(
                [
                    np.array(
                        [
                            np.power(
                                np.divide(
                                    np.linalg.norm(
                                        self.__data - self.__centers[j], axis=-1
                                    ),
                                    np.linalg.norm(
                                        self.__data - self.__centers[k], axis=-1
                                    ),
                                ),
                                2 / (self.m - 1),
                            )
                            for k in range(self.C)
                        ]
                    ).sum(axis=0)
                    for j in range(self.C)
                ],
                axis=-1,
            ),
        )

    def __centroids_unchanged(self):
        differences = np.abs(np.subtract(self.__centers, self.__previous_centers))
        return (differences <= self.tolerance).sum() == np.prod(differences.shape)

    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__features = self.__data.shape[1]
        self.__w = np.random.rand(self.__no_samples, self.C)
        self.__centers = np.empty(shape=(self.C, self.__features))

        t1 = time()
        for i in range(self.max_iter):
            self.__previous_centers = self.__centers.copy()
            self.__update_centroids()
            self.__update_w()
            if self.__centroids_unchanged():
                print(f"Algorithm stopped at iteration: {i}")
                break

        t2 = time()
        print(f"CMeans(iterative) time = {t2-t1}")


df = pd.read_csv(
    "data/BDA/Dataset 1/Iris-150.txt",
    header=None,
    names=["c1", "c2", "c3", "c4", "label"],
)
print(df.head())
df = df.drop("label", axis=1)
X = df.to_numpy()


t = time()
model = KMeans(
    n_clusters=3,
    n_init=10,
    max_iter=100,
    tol=0.001,
    init="random",
).fit(X)
t = time() - t
print(model.cluster_centers_)
print(model.cluster_centers_.shape)
print(model._n_threads)
print(model.inertia_)
print(model.labels_)
print(t)
t = time()
model = CMeans(C=3, m=2, max_iter=100, tolerance=0.001)
model.fit(X)
t = time() - t
centers = model.get_centers()
print(centers)
print(centers.shape)
print(t)
