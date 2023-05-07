import json
import pickle
import itertools

N1 = 2
N2 = 2
A = 1

filename = "all_feasible_MUL_2x2x1.dat"

def get_clust_from_aux(aux):
    """Returns a clustering from the provided auxiliary array"""
    clusts = []
    for a in [-1,1]:
        clusts.append(tuple([i for i, elem in enumerate(aux) if elem == a]))

    return tuple(clusts)

def main():
    clust_hash_table = {}
    with open('data/' + filename) as file:
        for line in file:
            aux = json.loads(line)[0]
            clustering = get_clust_from_aux(aux) 
            print(clustering)
            clust_hash_table[hash(clustering)] = clustering

    hash_file = open('data/hash_clusters_from_' + filename, 'wb')
    pickle.dump(clust_hash_table, hash_file)
    hash_file.close()

if __name__ == "__main__":
    main()
