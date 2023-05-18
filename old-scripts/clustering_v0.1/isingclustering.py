import numpy as np
import spinspace
import matplotlib.pyplot as plt
import ising
import math
from mathtools import trinum, ditri_array


# data structure for vector in p-space
class pvec:
    def __init__(self, h, J, spin=[]):
        if isinstance(h, np.ndarray):
            h = h.tolist()
        if isinstance(J, np.ndarray):
            J = J.tolist()

        print(h)
        print(J)
        self.gsize = len(h)
        self.vec = ditri_array(h, J)

    # converts index from upper triangular matrix to pvec
    def ijtok(self, i, j):
        if j < i:
            raise ValueError("Received (i,j) = {}! Cannot have j < i.".format((i, j)))

        if i >= self.gsize:
            raise ValueError(
                "Invalid index! First index {} larger than {}".format(i, self.gsize)
            )
        if j >= self.gsize:
            raise ValueError(
                "Invalid index! The second index {} is larger than {}".format(
                    j, self.gsize
                )
            )
        return trinum(self.gsize) - trinum(self.gsize - i) + (j - i)

    # converts index from pvec to (i,j)
    def ktoij(self, k):
        if k >= trinum(self.gsize):
            raise ValueError("Index out of range!")

        # inverting ijtok
        # just do the math
        triN = trinum(self.gsize)
        t = triN - k
        # this is the m such that trinum(m) = trinum(N - i)
        m = (math.isqrt(8 * t) + 1) // 2
        i = self.gsize - m
        j = trinum(m) - t + i

        return (i, j)

    # overload for []
    def __getitem__(self, index):
        # if index is a single integer
        if isinstance(index, int):
            return self.vec[index]

        # otherwise it is not and must be a tuple
        # ensure index is a tuple
        if not isinstance(index, tuple):
            raise TypeError("Expected integer or tuple, received {}".format(index))

        # ensure either one or two indicies are passed
        if len(index) not in [1, 2]:
            raise TypeError(
                "Expected tuple of length 1 or 2, received {}".format(index)
            )

        return self.vec[self.ijtok(index[0], index[1])]

    def validate(self, a, b):
        print("-------------")
        print(" Validation")
        print("-------------")
        vis = input("Visualize? (y/n)") in ["Y", "y"]

        for k in range(len(self.vec)):
            if self.vec[k] != self[self.ktoij(k)]:
                print("ERROR! Index k = {}".format(k))

        for i in range(self.gsize):
            for j in range(i, self.gsize):
                if self.vec[self.ijtok(i, j)] != self[i, j]:
                    print("ERROR! Index i,j = {}".format((i, j)))

        if vis:
            upper = pvec.ditri_to_numpy(a, b)
            test = self.tomatrix()
            print("EXPECTED:")
            print(upper)
            print("\nRECEIVED:")
            print(test)
            print("\n-----------------------\n")

    def toarray(self):
        return self.vec

    def tonumpy(self):
        return np.array(self.vec)

    def tomatrix(self):
        # easier notation for size
        N = self.gsize

        # index convenience function
        ind = lambda N, i, j: trinum(N) - trinum(N - i) + j

        # the numpy array
        test = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                test[i, j] = self.vec[ind(N, i, j - i)]

        return test

    @staticmethod
    def spinmin(spin):
        size = len(spin)
        h = -1 * np.array(spin)
        J = -1 * np.array(
            [-1 * spin[i] * spin[j] for i in range(size) for j in range(i + 1, size)]
        )
        return pvec(h=h, J=J)

    @staticmethod
    def pspin(spin):
        size = len(spin)
        h = np.array(spin)
        J = np.array(
            [spin[i] * spin[j] for i in range(size) for j in range(i + 1, size)]
        )
        return pvec(h=h, J=J)

    @staticmethod
    def ditri_to_numpy(di, tri):
        N = len(di)
        if len(tri) != trinum(N - 1):
            raise ValueError("Length error! Length of tri not compatible with di")

        l = lambda N, i: trinum(N) - trinum(N - i) - i
        d = lambda N, i: (N - 1) - i

        upper = np.zeros((N, N))
        ind = lambda N, i, j: trinum(N) - trinum(N - i) + j
        for i in range(N):
            x = l(N, i)
            y = x + d(N, i)
            upper[i, i:] = [di[i]] + tri[x:y]

        return upper


def ijtok(N, i, j):
    if j < i:
        raise ValueError("Received (i,j) = {}! Cannot have j < i.".format((i, j)))

    if i >= N:
        raise ValueError("Invalid index! First index {} larger than {}".format(i, N))
    if j >= N:
        raise ValueError(
            "Invalid index! The second index {} is larger than {}".format(j, N)
        )
    return trinum(N) - trinum(N - i) + (j - i)


# converts index from pvec to (i,j)
def ktoij(N, k):
    if k >= trinum(N):
        raise ValueError("Index out of range!")

    # inverting ijtok
    # just do the math
    triN = trinum(N)
    t = triN - k
    # this is the m such that trinum(m) = trinum(N - i)
    m = (math.isqrt(8 * t) + 1) // 2
    i = N - m
    j = trinum(m) - t + i

    return (i, j)


def check_index_conversion():
    # check that
    N = 10
    for i in range(N):
        for j in range(i, N):
            k = ijtok(N, i, j)
            print((i, j), "~~>", k, "~~>", ktoij(N, k))


def square_sample(dim, pmin, pmax):
    import random as rand

    return np.array([rand.random() * (pmax - pmin) + pmin for i in range(dim)])


def spherical_sample(dim, mu=0, sigma=1):
    import random as rand

    return np.random.normal(mu, sigma, dim)


# plot histogram of hamiltonian
def ham_distribution_on_cube(
    G: int,
    pmin: int = -10,
    pmax: int = 10,
    ndata: int = 100,
    minusnum: int = -1,
    normalize: bool = False,
    savefig=True,
    title="",
    show: bool = False,
    square: bool = True,
    mu: float = 0.0,
    sigma: float = 1.0,
):
    import random as rand

    mu, sigma = 0, 1
    figsize = (8, 4)

    # set figure dimensions
    fig = plt.figure()
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    # histogram settings
    nbins = 100
    width = 0.8
    box = [-1, 1]
    spin = []

    # randomly assign spins if no minusnum specified
    if minusnum == -1:
        spin = [rand.choice(box) for i in range(G)]

    else:
        spin = ([-1] * minusnum) + ([1] * (G - minusnum))

    # get the pspin of spin as numpy array
    pspin = pvec.pspin(spin).tonumpy()

    print("generating pvecs...", end="")
    # generate pvecs
    if square:
        X = np.array(
            [square_sample(dim=trinum(G), pmin=pmin, pmax=pmax) for i in range(ndata)]
        )
        title = "Distribution of energies for\nspin = {}\nwith h,J parameters uniformly chosen in [{},{}].\n {} pvecs sampled".format(
            spin, pmin, pmax, ndata
        )
    else:
        X = np.array(
            [spherical_sample(dim=trinum(G), mu=mu, sigma=sigma) for i in range(ndata)]
        )
        title = "Distribution of energies for\nspin = {}\nwith h,J parameters uniformly chosen on unit sphere. {} pvecs sampled".format(
            spin, ndata
        )
    print("done.")

    # normalize if applicable
    if normalize:
        print("normalizing pvecs...", end="")
        for i in range(len(X)):
            X[i, :] = X[i, :] / np.linalg.norm(X[i, :])
        print("done.")

    energy = np.array([])

    print("computing energies...")
    count = 0
    for x in X:
        if count % 1000 == 0:
            print("  on {} of {}".format(count, len(X)))
        energy = np.append(energy, np.dot(pspin, x))
        count += 1
    print("done.")

    counts, bins = np.histogram(energy)
    plt.hist(energy, nbins, rwidth=width)
    plt.title(title)
    plt.subplots_adjust(top=0.8)

    ####################
    # save figure prompt
    if savefig:
        if square:
            plt.savefig(
                "hamiltonian-square-dist/G={}_spin={}_range={}to{}".format(
                    G, spinspace.spin2dec(spin), pmin, pmax
                )
            )
        else:
            plt.savefig(
                "hamiltonian-square-dist/G={}_spin={}_spherical".format(
                    G, spinspace.spin2dec(spin), pmin, pmax
                )
            )
        print("saved figure")

    if show:
        plt.show()

    plt.clf()
    plt.close()


if __name__ == "__main__":
    import random as rand

    G = 16
    N = 100000
    pmin = -10
    pmax = 10
    show = False
    square = True
    normalize = False
    for c in range(G):
        ham_distribution_on_cube(
            G=G,
            pmin=pmin,
            pmax=pmax,
            minusnum=c,
            show=show,
            square=square,
            normalize=normalize,
            ndata=N,
        )
