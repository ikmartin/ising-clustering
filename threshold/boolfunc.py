# experiments to efficiently generate orbits of threshold functions under hyperoctahedral orbits
import math
from itertools import permutations


def spinactions(n):
    """Generator which iterates through elements of the spin action group (Z/2Z)^n."""
    for k in range(2**n):
        # each integer in range 0 - 2^n - 1 represents a spin action
        def func(num):
            # take bitwise XOR against input, equivalent to spin action
            return num ^ k

        yield func


def axisperms(n):
    """Generator which iterators through all axis permutations of size n. Each permutation takes an integer k as input and returns the integer obtained from permuting the bits of k."""
    for arr in permutations(list(range(n))):
        def func(k):
            b0 = bin(k)[2:].zfill(n)
            b1 = ''.join([b0[i] for i in arr])
            return int(b1, 2)

        yield func


def hyperoctahedral(n):
    """Generator which iterates through elements of the hyperoctahedral group. Each element is a callable which acts on both integers and on sequences of elements."""
    for perm in axisperms(n):
        for spin in spinactions(n):
            def func(k):
                return perm(spin(k))
            yield func

def func_action(dim, func, action):
    return [func[action(i)] for i in range(2**dim)]




class BoolFunc:
    def __init__(self, dim, index):
        """ Initializer of boolfunc
        Parameters
        ----------
        index : int
            the boolfunction as an integer
        dim : int
            the dimension of the domain of the boolfunction
        """
        self.dim = dim
        self.index=index
        self.vals=tuple(int(k) for k in bin(index)[2:].zfill(2**dim))

    def __getitem__(self, index):
        return self.vals[index]

    def __call__(self, index):
        return self.vals[index]

    def __str__(self):
        return str(self.index) + " <-> " + str(self.vals)

    def orbit(self):
        """Returns the orbit under the hyperoctahedral group of this BoolFunc"""
        dim = self.dim
        orbit = []
        for action in hyperoctahedral(dim):
            func = func_action(dim, self.vals, action)
            index = int(''.join([str(i) for i in func]),2)
            orbit.append(index)

        return set(orbit)

    def signed_orbit(self):
        """ Returns the orbit under the signed hyperoctahedral group of this BoolFunc"""
        inverter = (1<<(1<<self.dim)) - 1
        orbit = self.orbit()
        negorbit = set([ind ^ inverter for ind in orbit])
        return orbit.union(negorbit)

    def dual(self):
        vals = [not self.vals[~a] for a in range(1 << self.dim)]
        return BoolFunc(self.dim, vals)

    def selfdual(self):
        dvals = tuple([int(not self.vals[~a]) for a in range(1 << self.dim)])
        return BoolFunc(self.dim + 1, self.vals + dvals)

    @staticmethod
    def from_func(vals):
        d=int(math.log2(len(vals)))
        index=int(''.join([str(i) for i in vals]), 2)
        return BoolFunc(d, index)

    @staticmethod
    def random(dim):
        import random as rand
        return BoolFunc(dim, rand.randint(0, 2**(2**dim)-1))


####################################
### TESTS
####################################

def test_hyperoctahedral():
    d=3
    for j, func in enumerate(hyperoctahedral(d)):
        orig=[i for i in range(2**d)]
        perm=[func(i) for i in range(2**d)]
        print(j)
        print(" ", orig)
        print(" ", perm)

def test_orbit():
    import random
    dim = 3
    index = random.randint(0,2**(2**dim) - 1)
    bfunc = BoolFunc(dim, index)
    orbit = bfunc.signed_orbit()
    print(len(orbit), len(list(set(orbit))))
    check = len(orbit) == len(list(set(orbit)))
    print("There are no repeated elements in this orbit: ", check)
    print("Original:")
    print(" ",bfunc.vals)
    print("Orbit:")
    for k, ind in enumerate(orbit):
        temp = BoolFunc(dim, ind)
        print(k, " ", temp.vals)

if __name__ == "__main__":
    test_orbit()
