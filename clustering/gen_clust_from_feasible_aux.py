import json
import pickle
import itertools


def get_clust_from_aux(aux):
    """Returns a clustering from the provided auxiliary array"""
    clusts = []
    for a in [-1, 1]:
        clusts.append(tuple([i for i, elem in enumerate(aux) if elem == a]))

    return tuple(clusts)


def main():
    import os

    # gather parameters
    N1 = int(input("Enter N1: "))
    N2 = int(input("Enter N2: "))
    filename = str(input("Enter path to feasible aux file: "))

    clust_hash_table = {}
    with open(filename) as file:
        for line in file:
            aux = json.loads(line)[0]
            clustering = get_clust_from_aux(aux)
            print(clustering)
            clust_hash_table[hash(clustering)] = clustering

    hash_file = open("data/hash_clusters_from_IMul" + os.path.basename(filename), "wb")
    pickle.dump(clust_hash_table, hash_file)
    hash_file.close()


if __name__ == "__main__":
    main()
