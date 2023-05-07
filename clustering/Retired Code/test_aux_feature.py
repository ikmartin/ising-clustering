import ising
from spinspace import Spin

aux = [[-1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1]]

G = ising.IMul(2, 2)
G.set_aux(aux)
spin = G.inspace.getspin(1)
print("out", G.fout(spin).spin())
print("aux", G.faux(spin).spin())
print("out + aux", G.f(inspin=spin).spin())
print("in + out + aux", G.inout(inspin=spin).spin())

from topdown import TopDownSgnFarthestPair, TopDownQvecFarthestPair

ClustModel = TopDownSgnFarthestPair(circuit=G)
ClustModel.model()


def example():
    """NOTES
    This method of clustering breaks ties between clusters by attempting to minimize the average distance between points within clusters. It uses sgn as its ham_vec rather than qvec.

    It always terminates with 7 clusters, which is quite bad.
    """
    success = "\u2713"
    failure = "x"
    circuit = ising.IMul(2, 2)
    circuit.set_aux(aux)
    model = TopDownSgnFarthestPair(circuit=circuit, weak=True)
    model.model()

    while len(model.clusters) > 4:
        model = TopDownSgnFarthestPair(circuit=circuit)
        model.model()
        print("Clusters: ", len(model.clusters))

    clusters = model.clusters
    print(
        "RESULT OF TopDownFarthestPair on Mul2x2\n---------------------------------------"
    )
    print(f"Number of generations: {model.gen_num}")
    print(f"Final number of clusters: {len(model.clusters)}")

    for i, cluster in enumerate(clusters):
        print(f" cluster {i}:", cluster.indices)

    print("Generation progression:\n----------------------")
    for gen in model.generations:
        print(" ", len(gen))
        for cluster in gen:
            print("   ", cluster.indices, "  center:", cluster.center, end="")
            print("   ", success if cluster.satisfied else failure)


def example2():
    """NOTES
    This method of clustering breaks ties between clusters by attempting to minimize the average distance between points within clusters. It uses sgn as its ham_vec rather than qvec.

    It always terminates with 7 clusters, which is quite bad.
    """
    success = "\u2713"
    failure = "x"
    circuit = ising.IMul(2, 2)
    circuit.set_aux(aux)
    model = TopDownQvecFarthestPair(circuit=circuit, weak=True)
    model.model()

    while len(model.clusters) > 4:
        model = TopDownSgnFarthestPair(circuit=circuit)
        model.model()
        print("Clusters: ", len(model.clusters))

    clusters = model.clusters
    print(
        "RESULT OF TopDownFarthestPair on Mul2x2\n---------------------------------------"
    )
    print(f"Number of generations: {model.gen_num}")
    print(f"Final number of clusters: {len(model.clusters)}")

    for i, cluster in enumerate(clusters):
        print(f" cluster {i}:", cluster.indices)

    print("Generation progression:\n----------------------")
    for gen in model.generations:
        print(" ", len(gen))
        for cluster in gen:
            print("   ", cluster.indices, "  center:", cluster.center, end="")
            print("   ", success if cluster.satisfied else failure)


example()
