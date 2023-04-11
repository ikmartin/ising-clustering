from spinspace import Spinspace, Spinmode, Spin
from abc import ABC, abstractmethod
import spinspace as ss
import numpy as np
import math
from mathtools import ditri_array, trinum


# data structure for vector in p-space
class vspin:
    """Data structure for a vector in virtual space"""

    def __init__(self, h, J, spin=[]):
        if isinstance(h, np.ndarray):
            h = h.tolist()
        if isinstance(J, np.ndarray):
            J = J.tolist()

        print(h)
        print(J)
        self.gsize = len(h)
        self.vec = ditri_array(h, J)

    # converts index from upper triangular matrix to vspin
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

    # converts index from vspin to (i,j)
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

    # overload for indexing
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

    def __mul__(self, other):
        return sum([self.vec[i] * other[i] for i in range(self.vec)])

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
            upper = vspin.ditri_to_numpy(a, b)
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
        return vspin(h=h, J=J)

    @staticmethod
    def pspin(spin):
        size = len(spin)
        h = np.array(spin)
        J = np.array(
            [spin[i] * spin[j] for i in range(size) for j in range(i + 1, size)]
        )
        return vspin(h=h, J=J)

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


class PICircuit:
    """Base class for Pre Ising Circuits"""

    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M
        self.inspace = Spinspace(tuple([self.N]))
        self.outspace = Spinspace(tuple([self.M]))
        self.spinspace = Spinspace((self.N, self.M))
        self._graph = None

    # properties
    @property
    def graph(self):
        if self._graph == None:
            self._graph = self.generate_graph()
        return self._graph

    def energy(self, spin: Spin, ham_vec: np.ndarray):
        return np.dot(spin.vspin().spin(), ham_vec)

    @abstractmethod
    def f(self, inspin: Spin) -> Spin:
        pass

    def inout(self, inspin: Spin):
        """Returns the (in, out) pair corresponding to (s,f(s)) for an input spin s"""
        return Spin.catspin((inspin, self.f(inspin)))

    def generate_graph(self):
        graph = [self.inout(s) for s in self.inspace]
        return graph

    def level(self, inspin, ham_vec, more_info=False):
        """Returns information about the level"""

        # bool describing whether this level is satisfied by ham_vec
        satisfied = True

        # store info about the correct in/out pair
        s = inspin
        fs = self.f(inspin)
        correct_spin = self.inout(s)
        correct_key = correct_spin.asint()
        correct_energy = self.energy(correct_spin, ham_vec)

        # dictionary to store hamiltonian values
        ham = {correct_key: correct_energy}

        # iterate through the level
        for t in self.outspace:
            spin = Spin.catspin((s, t))
            energy = self.energy(spin=spin, ham_vec=ham_vec)

            if energy < correct_energy:
                satisfied = False
                if more_info == False:
                    break

            ham[spin.asint()] = energy

        if more_info:
            return satisfied, dict(sorted(ham.items(), key=lambda x: x[1]))
        else:
            return satisfied

    def levels(self, inspins: list[Spin], ham_vec, list_fails=False):
        """Returns information about the levels of a list of spins. By default, returns True if all levels are satisfied by ham_vec, False otherwise. If list_fails = True, then it returns a list of the input spins whose levels are not satisfied by ham_vec"""
        if list_fails:
            fails = []
            for s in inspins:
                if self.level(inspin=s, ham_vec=ham_vec) == False:
                    fails.append(s)
            return fails

        else:
            for s in inspins:
                if self.level(inspin=s, ham_vec=ham_vec) == False:
                    return False

            return True


class IMul(PICircuit):
    """Ising Multiply Circuit"""

    def __init__(self, N1: int, N2: int):
        super().__init__(N=N1 + N1, M=N1 + N2)
        self.inspace = Spinspace(shape=(N1, N2))
        self.N1 = N1
        self.N2 = N2

    def f(self, spin: Spin):
        # get the numbers corresponding to the input spin
        num1, num2 = spin.splitint()

        # multiply spins as integers and convert into spin format
        result = Spin(spin=num1 * num2, shape=(self.M,))
        return result


def example():
    G = IMul(2, 2)
    G.generate_graph()

    spin1 = G.inspace.rand()
    spin2 = G.inspace.rand()
    print(f"spin1 : {spin1.spin()}")
    print(f"spin1 : {spin2.spin()}")
    print(f"ham dist : {ss.dist(spin1,spin2)}")
    print(f"pairspin : {spin1.pairspin().spin()}")
    print(f"pairspin : {spin2.pairspin().spin()}")
    print(f"ham dist2 : {ss.dist2(spin1,spin2)}")
    print(f"vspin1 : {spin1.vspin(split=True).spin()}")
    print(f"vspin1 : {spin2.vspin(split=True).spin()}")
    print(f"vdist : {ss.vdist(spin1,spin2)}")
    print(f"pointwise multiply spin1 * spin2 : {ss.multiply(spin1,spin2)}")
    print(f"invert spin1 : {spin1.inv()}")


def example_Spin():
    spin1 = Spin(200, (4, 4))
    print(spin1.asint())
    print(spin1.spin())
    print(spin1.splitspin())
    print(spin1.splitint())


def level_test():
    G = IMul(2, 2)
    spin = G.inspace.getspin((2, 2))
    print(spin.spin())
    print(G.inout(inspin=spin).asint())
    print(G.inout(inspin=spin).spin())
    print(G.inout(inspin=spin).splitint())
    print(G.inout(inspin=spin).splitspin())
    print(G.inout(inspin=spin).pspin().spin())

    # test all pspins satisfy levels
    for s in G.inspace:
        ham_vec = G.inout(s).pspin().spin()
        sat, energies = G.level(inspin=s, ham_vec=ham_vec, more_info=True)
        print(f"Input level {s.asint()} is satisfied: {sat}")
        if sat == False:
            print("Examining error....")
            for key, value in energies.items():
                print(f"spin {key}  energy {value}")

    # test that it fails when expected
    ham_vec = G.inout(inspin=spin).pspin().spin()
    print(f"using pspin from input {spin.splitint()} to test levels.")
    for s in G.inspace:
        print(
            f"Level {s.splitint()} ~ {s.asint()} is {G.level(inspin=s, ham_vec=ham_vec)}"
        )


if __name__ == "__main__":
    level_test()
    # example()
    # example_Spin()
