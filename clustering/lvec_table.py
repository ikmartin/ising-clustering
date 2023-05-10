from ising import IMul
import numpy as np


def cosine_sim(vec1, vec2):
    """Translated cosine similarity, max value 2 min value 0 orthogonal at 1.

    Due to precision errors, np.dot(vec1,vec2)/norm(vec1)norm(vec2) will sometimes be slightly larger than 1 if vec1 == vec2. We manually check for this case.
    """
    if np.array_equal(vec1, vec2):
        return 0

    return (
        np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        / np.pi
    )


def gen_lvec_table_dist(circuit):
    N = circuit.inspace.size
    lvec = [
        circuit.normlvec(s) / np.linalg.norm(circuit.normlvec(s))
        for s in circuit.inspace
    ]
    A = np.array(
        [[np.linalg.norm(lvec[i] - lvec[j]) for i in range(N)] for j in range(N)]
    )
    return A


def gen_lvec_table_cosine(circuit):
    N = circuit.inspace.size
    lvec = [circuit.normlvec(s) for s in circuit.inspace]
    A = np.array([[cosine_sim(lvec[i], lvec[j]) for i in range(N)] for j in range(N)])
    return A


def main_dist():
    circuit = IMul(2, 3)
    np.savetxt(
        "data/lvec_table_dist_IMul2x3x0.csv",
        X=gen_lvec_table_dist(circuit),
        delimiter=",",
    )


def main():
    circuit = IMul(2, 3)
    np.savetxt(
        "data/lvec_table_cosine_IMul2x3x0.csv",
        X=gen_lvec_table_cosine(circuit),
        delimiter=",",
    )


if __name__ == "__main__":
    main()
    main_dist()
