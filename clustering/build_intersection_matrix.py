import numpy as np
from ising import IMul
import time


def build_intersection(circuit: IMul, repetitions=50000, deg=2, filename=""):
    # build default filename
    if filename == "":
        filename = f"data/input_level_intersection_IMul{circuit.N1}x{circuit.N2}x{circuit.A}_rep={repetitions}_deg={deg}.csv"
    # mean and variance for gaussian random hJ vecs
    mu, sigma = 0, 1

    # store the number of distinct input values
    N = circuit.inspace.size

    # initialize empty matrix prior to starting main loop, save start time
    matrix = np.zeros(tuple(N for _ in range(deg)))
    time0 = time.time()
    for i in range(repetitions):
        # get the hJ random vector
        hvec = np.random.normal(mu, sigma, circuit.hJ_num)

        # get the results from levels and store length
        results = circuit.levels(ham_vec=hvec, detailed=True)

        # recursively take outer products
        newres = results
        for j in range(2, deg + 1):
            newres = np.outer(newres, results).reshape(tuple(N for _ in range(j)))

        # add to matrix
        matrix += newres

        # update to console
        if i % 500 == 0:
            print(f"Completed {i} iterations in {time.time() - time0} sec")

    # normalize
    matrix = matrix / repetitions

    np.savetxt(fname=filename, X=matrix.reshape(N ** (deg - 1), N), delimiter=",")

    return matrix


def interface(circuit):
    import time

    # record start time
    t0 = time.time()

    deg = int(input("Enter the degree intersection: "))
    repetitions = int(input("Enter the number of iterations to run: "))
    X = build_intersection(circuit, repetitions=repetitions, deg=deg)
    print(X)

    # record final time and display
    print(f"Computed {repetitions} repetitions in {time.time() - t0} seconds")


if __name__ == "__main__":
    interface(IMul(2, 3))
