from time import monotonic
import os

import logging
logging.basicConfig(level=logging.DEBUG)

import nmslib
from numpy import save
from clustering import get_layers
from data import Dataset

METHOD_NAMES = ("hcnsw", "hnsw")
#DB_NAME = 'sift1m'
DB_NAME = 'gist'
#DB_NAME = 'glove-50'
SPACE = 'l2'
#SPACE = 'angulardist'
SEED_RANGE = range(1, 31)

dataset = Dataset()
print(f"loading {DB_NAME}...")
dataset.load(DB_NAME) 
print(f"loading {DB_NAME} done")

def cluster_to_file(dataset, m, seed):
    filename = f"data/levels_{DB_NAME}_seed{seed}.txt"
    if os.path.isfile(filename):
        return
    layers = get_layers(dataset.get_db_embeddings(), m, seed=seed)
    with open(filename, 'w') as file:
        for layer in layers:
            file.write(str(layer) + '\n')
    print(f"saved to {filename}")

def indexing(dataset, method, seed=0):
    m = 20
    if method == 'hcnsw':
        index_time_params = {'M': m, 'indexThreadQty': 1, 'efConstruction': 100, 'post' : 0, 'levels_file': f"data/levels_{DB_NAME}_seed{seed}.txt", 'delaunay_type' : 0}
    elif method == 'hnsw':
        index_time_params = {'M': m, 'indexThreadQty': 1, 'efConstruction': 100, 'post' : 0, 'seed':seed, 'delaunay_type' : 0}
    else:
        raise Exception(f"Método <{method}> não compatível")
    index = nmslib.init(method=method, space=SPACE, data_type=nmslib.DataType.DENSE_VECTOR) 
    index.addDataPointBatch(dataset.get_db_embeddings())
    if method == 'hcnsw':
        print(f"clustering w/ seed={seed} ...")
        cluster_to_file(dataset, m, seed) 
        print(f"clustering done")
    print(f"indexing...")
    index.createIndex(index_time_params)
    print(f"indexing done")
    #index.saveIndex(f'data/{method}_d0_{DB_NAME}_seed{seed}.bin', save_data=False)
    return index

def load_index(method, seed):
    index = nmslib.init(method=method, space=SPACE, data_type=nmslib.DataType.DENSE_VECTOR) 
    print(f"loading index...")
    index.loadIndex(f'data/{method}_d0_{DB_NAME}_seed{seed}.bin')
    print(f"loading index done")
    return index

def test(index, dataset, k=100):
    query_time_params = {'efSearch': 100}
    index.setQueryTimeParams(query_time_params)
    recalls = []
    deltas = []
    for i in range(dataset.get_test_size()):
        query = dataset.get_test_embedding(i)
        start = monotonic()
        ann_results = index.knnQuery(query, k=k)
        end = monotonic()
        recall = dataset.get_test_recall(i, ann_results)
        recalls.append(recall)
        delta = end-start
        deltas.append(delta)
        print(f"test {i}: recall@{k}={recall} time={round(delta, 6)}")
    avg_recalls = round(sum(recalls)/len(recalls), 4)
    avg_deltas = round(sum(deltas)/len(deltas), 6)
    print(f"Final results (avg): recall@{k}={avg_recalls} time={avg_deltas}")
    return recalls, deltas

for seed in SEED_RANGE:
    for method in METHOD_NAMES:
        index = indexing(dataset, method, seed)
        #index = load_index(method, seed)
        recalls, deltas = test(index, dataset)
        save(f"data/{method}_d0_{DB_NAME}_seed{seed}_recalls.npy", recalls)
        save(f"data/{method}_d0_{DB_NAME}_seed{seed}_deltas.npy", deltas)
        index = None
        
    # hcnsw (seed 1, delaunay 0): Final results (avg): recall@100=0.9071 time=-0.006317
    # hnsw (seed 1, delaunay 0): Final results (avg): recall@100=0.88 time=0.000147
