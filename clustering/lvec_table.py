from ising import IMul
import numpy as np


def gen_lvec_table(circuit):
    N = circuit.inspace.size
    lvec = [circuit.normlvec(s) for s in circuit.inspace]
    A = np.array(
        [[np.linalg.norm(lvec[i] - lvec[j]) for i in range(N)] for j in range(N)]
    )
    return A


def main():
    circuit = IMul(2, 2)
    np.savetxt(
        "data/lvec_table_IMul2x2x0.csv", X=gen_lvec_table(circuit), delimiter=","
    )


if __name__ == "__main__":
    main()
