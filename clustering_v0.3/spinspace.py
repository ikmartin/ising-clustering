import numpy as np

#########################################
### Spin Space Methods
#########################################


def trinum(n):
    """Returns the nth triangular number"""
    return int(n * (n + 1) / 2)


def int2spin(val: int, dim: int) -> np.ndarray:
    """Generate spin representation as a numpy.ndarray of integer.

    Parameters
    ----------
    val : int
        the integer value to convert
    dim : int
        the dimension of the spinspace where this spin lives

    Returns
    -------
    numpy.ndarray
        a 1-d array consisting of -1 and 1 representing a spin
    """

    b = list(np.binary_repr(val).zfill(dim))  # get binary representation of num
    a = [-1 if int(x) == 0 else 1 for x in b]  # convert to spin representation
    return np.array(a).astype(np.int8)  # return as a numpy array


def spin2int(spin: np.ndarray):
    """Generate integer representation of a spin

    Parameters
    ----------
    spin : numpy.ndarry or tuple of numpy.ndarray

    Returns
    -------
    int or tuple of int

    """

    # store the length of spin
    N = len(spin)

    # number to return
    num = tuple([2 ** (N - (i + 1)) * (1 if spin[i] == 1 else 0) for i in range(N)])
    return sum(num)


class Spin:
    """Represents a single element of a Spinspace

    Attributes
    ----------
    val : int
        the integer which this spin represents in binary
    shape : tuple(int)
        the shape of the spin, (2,3) means this is decomposed into S^2 x S^3; (5,) means this is decomposed into S^5

    Methods
    -------
    asint() -> int:
        returns the integer value of this spin
    spin() -> ndarray:
        returns the spin as an ndarray of -1 and 1's
    splitspin() -> tuple(ndarray)
        returns the spin as a tuple of ndarrays in the shape of self.shape
    splitint() -> tuple(int)
        returns the spin as a tuple of integers in the shape of self.shape
    """

    def __init__(self, spin, shape: tuple):
        """Initializer"""
        # store the shape first
        self.shape = shape

        # different methods to retreive self.val depending on the input type of spin
        if isinstance(spin, int):
            # if provided spin is an integer, just store value
            # e.g. spin = 9
            self.val = spin
        elif isinstance(spin, np.ndarray):
            # if provided spin is an ndarray, convert to int
            # e.g. spin = [ 1,-1,-1, 1]
            self.val = spin2int(spin)
        elif isinstance(spin, tuple):
            # additional checks if spin is passed as a tuple
            # depending on whether it is a splitspin or a splitint
            if isinstance(spin[0], np.ndarray):
                # if it is splitspin concatenate and then convert
                # e.g. spin = ([ 1,-1], [-1, 1])
                self.val = spin2int(np.concatenate(spin))
            elif isinstance(spin[0], int):
                # if it is splitint
                # e.g. spin = (2, 1)
                tempspin = tuple(int2spin(s, self.shape[i]) for i, s in enumerate(spin))
                self.val = spin2int(np.concatenate(tempspin))
            else:
                raise Exception("val not initialized")
        else:
            raise Exception("val not initialized")

    def __getitem__(self, key):
        return self.spin()[key]

    def asint(self):
        """Return the integer value represented by this spin in binary"""
        return self.val

    def spin(self):
        """Return this spin as an array of 1's and -1's"""
        return int2spin(val=self.val, dim=sum(self.shape))

    def splitspin(self):
        """Get this spin in split-spin format"""

        # convenience variables
        dim = sum(self.shape)
        indices = [sum(self.shape[:i]) for i in range(1, len(self.shape))]

        return np.split(int2spin(self.val, dim=dim), indices)

    def splitint(self):
        # convenience variables
        dim = sum(self.shape)
        indices = [sum(self.shape[:i]) for i in range(1, len(self.shape))]

        tempspin = self.splitspin()
        return tuple(spin2int(s) for s in tempspin)

    def dim(self):
        """Returns the dimension of the spin space in which this Spin lives"""
        return sum(self.shape)

    def pairspin(self):
        """Returns the spin corresponding to the pairwise interactions of spin."""
        # TODO: ensure the spin is an instance of Spin

        # store the spin representation
        a = self.spin()
        dim = self.dim()

        # iterate through the interactions
        pair = []
        for i in range(dim):
            for j in range(i + 1, dim):
                pair.append(a[i] * a[j])
        return Spin(spin2int(np.array(pair)), shape=(len(pair),))

    def vspin(self, split=False):
        """Returns the spin in virtual spinspace corresponding to this spin."""
        pair = self.pairspin()

        if split:
            return Spin(
                spin=(self.asint(), pair.asint()), shape=(self.dim(), pair.dim())
            )
        else:
            spin = Spin.catspin((self, pair))
            return Spin(spin=spin.asint(), shape=(self.dim() + pair.dim(),))

    def pspin(self, split=False):
        return self.vspin(split=split).inv()

    def inv(self):
        """Returns the multiplicative inverse of the provided spin.

        Returns: (numpy.ndarray)

        Params:
        *** s: (numpy.ndarray) the spin to invert
        """
        s = self.spin()
        return Spin(spin=np.array([-1 * si for si in s]), shape=self.shape)

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return self.__str__()

    ############################################
    ### Static Methods
    ############################################

    @staticmethod
    def catspin(spins: tuple):
        shape = sum(tuple(s.shape for s in spins), ())
        val = sum(tuple(s.splitint() for s in spins), ())
        return Spin(spin=val, shape=shape)

    @staticmethod
    def lin_comb(spins: list):
        """Takes a 'linear combination' of spins by summing and then taking sign.
        Requires an odd number of spins to guarentee nonzero entries.
        """
        if len(spins) % 2 == 0:
            raise Exception(
                "Warning: cannot take the linear combination of an even number of spins!"
            )

        return Spin(spin=np.sign(qvec(spins)), shape=spins[0].shape)


class Spinspace:
    """
    Wrapper class for a spinspace of a specified size

    Attributes
    ----------
    shape : tuple
        the shape of the spinspace
    dim : int
        the dimension of the spinspace
    size : int
        the cardinality of the spinspace
    shape : int
        the shape taken by spins. (2,2) means S^2 x S^2, (4,) means S^4
    split : bool
        convenience flag, False if spinspace comprised of one component, True if decomposed into multiple components

    Methods
    -------
    __init__(shape : tuple)
        initializes spinspace. If not decomposing spinspace then set shape = (dim) where dim is the desired dimension
    __iter__()
        makes this class an iterator.
    __next__()
        fetches next element. Converts _current_index from an integer to a spin in the necessary format
    getspin(spin)
        convenience function, ensures that paramter spin is of type Spin, rather than an integer or an array or a tuple thereof
    rand()
        returns a random spin from this spinspace, uniformly sampled
    dist(spin1,spin2)
        wrapper for hamming distance
    dist2(spin1, spin2)
        wrapper for second order hamming distance
    vdist(spin1, spin2)
        wrapper for hamming distance in virtual spinspace
    """

    def __init__(self, shape: tuple):
        self.shape = shape
        self.dim = sum(shape)
        self.size = 2**self.dim
        self.split = False if len(shape) == 1 else True
        self._current_index = 0

    def __iter__(self):
        """Makes this object iterable"""
        return self

    def __getitem__(self, key):
        return self.getspin(key)

    def __next__(self):
        """Returns the next spin in the iteration formatted in the appropriate mode. Converts _current_index to the appropriate spin."""
        if self._current_index >= self.size:
            self._current_index = 0
            raise StopIteration

        # gets the spin corresponding to _current_index
        spin = Spin(self._current_index, shape=self.shape)
        self._current_index += 1

        return spin

    def getspin(self, spin) -> Spin:
        """Convenience function, ensures the argument passed is of type Spin and not just an array or an integer representing a spin.
        Parameters
        ----------
        spin
            the thing to ensure is of type Spin

        Returns
        -------
        an instance of Spin
        """
        if isinstance(spin, Spin) == False:
            spin = Spin(spin=spin, shape=self.shape)
        return spin

    def tolist(self):
        """Returns a list of integers representing this spin space"""
        return list(range(self.size))

    def rand(self):
        """Returns a random spin from this spinspace, sampled uniformly"""
        from random import randint

        a = randint(0, self.size - 1)
        return Spin(a, self.shape)

    def dist(self, spin1, spin2):
        """Return the hamming distance between two spins. Easiest to first convert to spin mode"""
        if isinstance(spin1, Spin) == False:
            spin1 = Spin(spin1, shape=self.shape)

        if isinstance(spin2, Spin) == False:
            spin1 = Spin(spin2, shape=self.shape)

        return sum(np.not_equal(spin1.spin(), spin2.spin()))

    def dist2(self, spin1, spin2):
        """Return the 2nd order hamming distance between two spins. Not efficiently implemented."""
        if isinstance(spin1, Spin) == False:
            spin1 = Spin(spin1, shape=self.shape)

        if isinstance(spin2, Spin) == False:
            spin1 = Spin(spin2, shape=self.shape)

        s1, s2 = spin1.spin(), spin2.spin()

        if len(s1) != len(s2):
            raise Exception(
                f"Error: len(spin1) : {len(s1)} is not equal to len(spin2) : {len(s2)}"
            )

        result = 0
        for i in range(len(s1)):
            for j in range(i + 1, len(s1)):
                result += bool(s1[i] * s1[j] - s2[i] * s2[j])

        return result

    def vdist(self, spin1, spin2):
        """Returns the distance between spin1 and spin2 in virtual space"""
        return self.dist(spin1, spin2) + self.dist2(spin1, spin2)


##################################################################
## OTHER FUNCTIONS IN THIS MODULE
##################################################################


def dist(spin1: Spin, spin2: Spin):
    if len(spin1.spin()) != len(spin2.spin()):
        raise Exception(
            f"Error: len(spin1) : {len(spin1.spin())} is not equal to len(spin2) : {len(spin2.spin())}"
        )

    return sum(np.not_equal(spin1.spin(), spin2.spin()))


def dist2(spin1: Spin, spin2: Spin):
    """Return the 2nd order hamming distance between two spins. Not efficiently implemented."""
    if len(spin1.spin()) != len(spin2.spin()):
        raise Exception(
            f"Error: len(spin1) : {len(spin1.spin())} is not equal to len(spin2) : {len(spin2.spin())}"
        )

    s1 = spin1.spin()
    s2 = spin2.spin()

    result = 0
    for i in range(len(s1)):
        for j in range(i + 1, len(s2)):
            result += bool(s1[i] * s1[j] - s2[i] * s2[j])

    return result


def vdist(spin1: Spin, spin2: Spin):
    """Returns the distance between spin1 and spin2 in virtual space"""
    return dist(spin1, spin2) + dist2(spin1, spin2)


def diameter(spins: list[Spin]) -> int:
    l = len(spins)
    maxdist = 0
    for i in range(l):
        for j in range(i + 1, l):
            d = vdist(spins[i], spins[j])
            maxdist = d if d > maxdist else maxdist

    return maxdist


def qvec(spins):
    vec = -1 * np.sum([spin.vspin().spin() for spin in spins], axis=0)
    return vec


def sgn(spins):
    """Takes the sign of a bunch of spins. Wrapper for numpy.sign which can take list[Spin] as input."""
    return np.sign(qvec(spins))


def hamming(arr1: np.ndarray, arr2: np.ndarray):
    """Returns hamming distance between two numpy.ndrray of equal length"""
    return sum([1 if arr1[i] == arr2[i] else 0 for i in range(len(arr1))])


def multiply(s: Spin, t: Spin):
    """Multiplies two spins of equal length

    Returns: (numpy.ndarray) numpy 1D array of length equal to length of inputs. Entries are 1's and -1's.

    Params:
    *** s: (numpy.ndarray) the first spin
    *** t: (numpy.ndarray) the second spin
    """

    if s.dim() != t.dim():
        raise ValueError("Lengths of spins don't match, cannot multiply!")

    return np.multiply(s.spin(), t.spin())
