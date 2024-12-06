from time import time

import nmslib
from data import Dataset

DB_NAME = 'sift1m'
dataset = Dataset()
print(f"loading {DB_NAME}...")
dataset.load_from_tfds(DB_NAME) 
print(f"loading {DB_NAME} done")

def indexing(dataset):
    index_time_params = {'M': 20, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 0}
    index = nmslib.init(method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR) 
    index.addDataPointBatch(dataset.get_db_embeddings()) 
    print(f"indexing...")
    index.createIndex(index_time_params)
    print(f"indexing done")
    index.saveIndex(f'data/hnsw_{DB_NAME}.bin', save_data=False)
    return index

def load_index():
    index = nmslib.init(method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR) 
    print(f"loading index...")
    index.loadIndex(f'data/hnsw_{DB_NAME}.bin')
    print(f"loading index done")
    return index

def test(index, dataset, k=100):
    query_time_params = {'efSearch': 100}
    index.setQueryTimeParams(query_time_params)
    recalls = []
    deltas = []
    for i in range(dataset.get_test_size()):
        query = dataset.get_test_embedding(i)
        start = time()
        ann_results = index.knnQuery(query, k=k)
        #print(ann_results)
        #input("a")
        end = time()
        recall = dataset.get_test_recall(i, ann_results)
        recalls.append(recall)
        delta = end-start
        deltas.append(delta)
        print(f"test {i}: recall@{k}={recall} time={round(delta, 4)}")
    avg_recalls = round(sum(recalls)/len(recalls), 4)
    avg_deltas = round(sum(deltas)/len(deltas), 4)
    print(f"Final results (avg): recall@{k}={avg_recalls} time={avg_deltas}")
    return recalls, deltas

#index = indexing(dataset)
index = load_index()

test(index, dataset)