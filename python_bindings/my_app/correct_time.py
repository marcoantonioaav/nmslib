from numpy import load, save
import nmslib
from data import Dataset
from time import monotonic

DB_NAMES = ('sift1m', 'gist', 'glove-50')
DB_SPACES = ('l2', 'l2', 'angulardist')
METHOD_NAMES = ("hnsw", "hcnsw")
SEED_RANGE = range(1, 31)

def load_dataset(db):
    dataset = Dataset()
    print(f"loading {db}...")
    dataset.load(db) 
    print(f"loading {db} done")
    return dataset

def load_index(method, space, db, seed):
    index = nmslib.init(method=method, space=space, data_type=nmslib.DataType.DENSE_VECTOR) 
    print(f"loading index...")
    index.loadIndex(f'data/{method}_{db}_seed{seed}.bin')
    print(f"loading index done")
    return index

for i_db, db in enumerate(DB_NAMES):
    space = DB_SPACES[i_db]
    dataset = None
    dataset = load_dataset(db)
    for method in METHOD_NAMES:
        for seed in SEED_RANGE:
            index = None
            index = load_index(method, space, db, seed)
            deltas = []
            old_deltas = load(f"data/{method}_{db}_seed{seed}_deltas.npy")
            for i in range(len(old_deltas)):
                if old_deltas[i] > 0:
                    deltas.append(old_deltas[i])
                else:
                    query = dataset.get_test_embedding(i)
                    start = monotonic()
                    ann_results = index.knnQuery(query, k=100)
                    end = monotonic()
                    delta = end-start
                    deltas.append(delta)
                    print(f"{method} {db} {seed} : old time ={round(old_deltas[i], 6)} new time ={round(delta, 6)}")
            save(f"data/{method}_{db}_seed{seed}_deltas.npy", deltas)
                