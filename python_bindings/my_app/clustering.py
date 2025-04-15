from numpy.linalg import norm
import random
from sklearn.cluster import BisectingKMeans

def get_distance(e1, e2):
    return norm(e1-e2) ## Euclidean distance (L2 norm)

def get_nearest(node_indexes_and_distances):
    if len(node_indexes_and_distances) == 0:
        return None
    result = node_indexes_and_distances[0]
    for index, distance in node_indexes_and_distances[1:]:
        if distance < result[1]:
            result = (index, distance)
    return result[0]

def get_layers(data, m:int, max_clusters=1024, seed=0):
    random.seed(seed)
    candidates = list(range(len(data)))
    layers = [None] * len(data)
    current_layer = 0
    layer_increment_size = 1
    while layer_increment_size <= max_clusters and len(candidates) > 0:
        kmeans = BisectingKMeans(n_clusters=layer_increment_size, bisecting_strategy='largest_cluster', random_state=seed)
        kmeans.fit(data)
        centroids = kmeans.cluster_centers_
        for centroid in centroids:
            candidate_distances = [(i, get_distance(data[i], centroid)) for i in candidates]
            best_candidate = get_nearest(candidate_distances)
            layers[best_candidate] = current_layer
            candidates.remove(best_candidate)
        print(f"layers selected: {len(data) - len(candidates)}")
        current_layer += 1
        layer_increment_size = min(m*layer_increment_size, len(candidates))

    while len(candidates) > 0:
        layer_candidates = random.sample(candidates, layer_increment_size)
        for layer_candidate in layer_candidates:
            layers[layer_candidate] = current_layer
            candidates.remove(layer_candidate)
        current_layer += 1
        print(f"layers selected: {len(data) - len(candidates)}")
        layer_increment_size = min(m*layer_increment_size, len(candidates))
    layers = [(current_layer-1)-l for l in layers]
    return layers