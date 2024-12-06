import numpy as np

np.random.seed(0)

class Dataset():
    def __init__(self) -> None:
        self.db = []
        self.test_embeddings = []
        self.test_neighbors = []
        self.pca_db = []

    def load_from_tfds(self, name):
        self.test_embeddings = np.load(f"data/tfds_test_embeddings_{name}.npy").tolist()
        self.test_embeddings = [np.array(i) for i in self.test_embeddings]
        self.test_neighbors = np.load(f"data/tfds_test_neighbors_{name}.npy").tolist()
        self.db = np.load(f"data/tfds_db_{name}.npy").tolist()
        self.db = [np.array(i) for i in self.db]
        self.pca_db = np.load(f"data/tfds_db_{name}2d.npy").tolist()
        self.pca_db = [np.array(i) for i in self.pca_db]

    def get_test_size(self):
        return len(self.test_embeddings)

    def get_db_embeddings(self):
        return self.db

    def get_test_embedding(self, test_index):
        return self.test_embeddings[test_index]

    def get_test_recall(self, test_index, search_result):
        k = len(search_result[0])
        gold_result = [i for i in self.test_neighbors[test_index][:k]]
        found = 0
        for i in range(k):
            if search_result[0][i] in gold_result:
                found += 1
        return found/k

def generate_embeddings(n, size=128):
    embeddings = []
    for i in range(n):
        embeddings.append(generate_embedding(size))
    return embeddings

def generate_embedding(size=128):
    return np.array(np.random.rand(size))