from clustering import get_layers
from data import Dataset

def cluster_to_file(dataset, m, seed, filename):
    layers = get_layers(dataset.get_db_embeddings(), m, seed=seed)
    with open(filename, 'w') as file:
        for layer in layers:
            file.write(str(layer) + '\n')
      
DB_NAME = 'sift1m'
dataset = Dataset()
print(f"loading {DB_NAME}...")
dataset.load_from_tfds(DB_NAME) 
print(f"loading {DB_NAME} done")

M = 20      

for seed in range(1, 31):
    filename = f"data/levels_{DB_NAME}_seed{seed}.txt"
    print(f"clustering w/ seed={seed} ...")
    cluster_to_file(dataset, M, seed, filename)
    print(f"done (saved to {filename})")