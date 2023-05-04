import math
from typing import Callable
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score

class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name, func: Callable):
        self._registry[name] = func

    def get(self, name):
        return self._registry.get(name)

def silhouette_score_wrapper(X_train, y_train, cluster_assignments_train):
    return silhouette_score(X_train, cluster_assignments_train)

def adjusted_rand_score_wrapper(X_train, y_train, cluster_assignments_train):
    if y_train is None:
        return math.nan
    return adjusted_rand_score(y_train, cluster_assignments_train)

def calinski_harabasz_score_wrapper(X_train, y_train, cluster_assignments_train):
    return calinski_harabasz_score(X_train, cluster_assignments_train)



metrics_registry = Registry()

metrics_registry.register('silhouette_score', silhouette_score_wrapper)
metrics_registry.register('adjusted_rand_score', adjusted_rand_score_wrapper)
metrics_registry.register('calinski_harabasz_score', calinski_harabasz_score_wrapper)
