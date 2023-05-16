from functools import cache
from copy import Error
from typing import Callable
from spinspace import Spinspace, Spin
from abc import abstractmethod
from ortools.linear_solver import pywraplp
import spinspace as ss
import numpy as np


class PICircuit:
    """Base class for Pre Ising Circuits.

    Comments
    --------
    Regarding auxiliary spins: There are two use-cases for auxiliary spins depending on whether you are attempting to check constraints equations or model dynamis;
        (1) Checking global constraint equations: in this case, auxiliary states are tied to specific inputs, and one is likely curious whether the choice of auxiliary state at each input level yields a solvable Ising circuit. This should be done by setting a feasible auxiliary array after initializing a circuit using the `set_all_aux` method.
        (2) Modeling dynamics: in this case, one likely wishes to include auxiliary vertices in an Ising graph without specifying which states correspond to each input. Indeed, there need not be a consistent `correct` auxiliary state at each input level since it is ignored in the final output. In this situation, one should set the optional parameter `A` in class initialization.
    Use case (1) is handled by the circuit logic in the functions `f` and `faux`. The latter function will raise an Attribute Use case (2) is handled by the attribute `auxspace`,

    IMPORTANT NOTE: Calling set_all_aux will not overwrite auxspace UNLESS the provided feasible auxiliary array is of shape larger than the current auxspace. The ambient auxspace can be larger than the set auxspins. For example,
        circuit.PICircuit(N=2,M=2,A=2)
        circuit.set_all_aux([[-1],[-1],[1],[1]])
    will result in a circuit whose auxspace has dimension 2, but whose `faux` method will return auxiliary spins of dimension 1.

    Attributes
    ----------
    self.N : int
        the number of input vertices in this circuit
    self.M : int
        the number of output vertices in this circuit
    self.A : int
        the number of auxiliary vertices in this circuit
    self.inspace : Spinspace
        a Spinspace of shape (N), contains all the input states of the circuit
    self.outspace : Spinspace
        a Spinspace of shape (M), contains all the output states of the circuit
    self.auxspace : Spinspace
        a Spinspace of shape (A), contains all the auxiliary states of the circuit.
        NOTE: does not necessarily match the dimension of spins returned by `faux`.
    self.spinspace : Spinspace
        the total spinspace of the graph, has shape (N, M, A).

    Properties
    ----------
    self.G : int
        returns the total number of vertices in this circuit, N + M + A
    self.hJ_num : int
        returns the #(h coordinates) + #(J coordinates) for this circuit. Equal to the Gth triangular number.
    self.graph : list[Spin]
        this is the `graph` of the logic of G, i.e. a list of Spins (input, fout(input), faux(input)).
    self.fulloutspace : Spinspace
        a spinspace of shape (M, A') where A' is the shape of the set feasible auxiliary states. Is called 'fulloutspace' since it contains output spins together with those auxiliary spins with a 'correct' value at each input.

    Methods
    -------

    """

    def __init__(self, N: int, M: int, A: int = 0):
        self.N = N
        self.M = M
        self.A = 0
        self._init_spaces()

    # convenient initialization thing
    def _init_spaces(self):
        self.inspace = Spinspace((self.N,))
        self.outspace = Spinspace((self.M,))
        self.auxspace = Spinspace((self.A,))
        self.spinspace = Spinspace((self.N, self.M, self.A))
        self._graph = None
        self._aux_dict = {}

    #############################################
    # Properties and property-like methods
    #############################################
    @property
    def G(self):
        return self.N + self.M + self.A

    @property
    def hJ_num(self):
        return int(self.G * (self.G + 1) / 2)

    @property
    def graph(self):
        if self._graph == None:
            self._graph = self._generate_graph()
        return self._graph

    #################################
    ### Private methods
    #################################

    def _generate_graph(self):
        graph = [self.inout(s) for s in self.inspace]
        return graph

    #################################
    ### AUXILIARY STATE METHODS
    #################################

    def set_all_aux(self, aux_array):
        """
        Sets the auxiliary array. Expects a list of lists. Intelligently converts provided input into a list of Spins.

        Parameters
        ----------
        aux_array : list[list]
            the auxiliary array to use for this PICircuit. Must be of consistent shape, either (2^N, A) or (A, 2^N) (in numpy notation).
        """
        # reinitialize _aux_array
        self._aux_dict = {}

        # convert to numpy array
        aux_array = np.array(aux_array)

        # check to ensure either rows or columns matches 2^N
        if aux_array.shape[0] != 2 ** self.N and aux_array.shape[1] != 2**self.N:
            raise ValueError("The aux_array must have one auxiliary state per input!")

        # we want the first index to correspond to the input, not to the coordinate of the aux state
        if aux_array.shape[1] == 2**self.N:
            aux_array = aux_array.T

        # record the number of set auxiliary spins and construct the necessary spinspaces
        self.A = len(aux_array[0])
        self._outauxspace = Spinspace(shape=(self.M, self.A))

        # check for consistent length and store aux_array as list of Spins
        for i in range(aux_array.shape[0]):
            row = aux_array[i]
            if len(row) != self.A:
                raise ValueError("Not all auxiliary states are the same length!")

            self._aux_dict[self.inspace[i]] = Spin(spin=row, shape=(self.A,))

        self._init_spaces()

    def get_aux_array(self):
        """Returns the aux array of this circuit"""
        return [self._aux_dict[inspin].spin().tolist() for inspin in self.inspace]

    ##################################
    ### CIRCUIT LOGIC METHODS
    ##################################

    # this method required to be overwritten in inherited classes
    @abstractmethod
    def fout(self, inspin: Spin) -> Spin:
        pass

    def faux(self, inspin: Spin) -> None | Spin:
        if self.A == 0:
            return None
        else:
            return self._aux_dict[inspin]

    def f(self, inspin: Spin | int) -> Spin:
        # ensure is of type Spin
        if isinstance(inspin, int):
            inspin = self.inspace.getspin(inspin)

        out = self.fout(inspin)
        aux = self.faux(inspin)
        if aux is None:
            return out
        else:
            return Spin(spin=(out.asint(), aux.asint()), shape=(self.M, aux.dim()))

    def inout(self, inspin: Spin):
        """Returns the (in, out) pair corresponding to (s,f(s)) for an input spin s. If a list of Spins is provided instead then a list of (in, out) pairs returned."""
        return Spin.catspin((inspin, self.f(inspin)))

    #################################
    ### GENERATORS
    #################################

    def inputlevelspace(self, inspin):
        """Generator which spits out all input/output/aux pairs with a fixed input and auxiliary of the correct size for the given input"""
        outauxspace = Spinspace(shape=(self.M, self.A))
        for outaux in outauxspace:
            yield Spin.catspin((inspin, outaux))

    def allwrong(self, inspin, tempaux=Spin(0, shape=(0,))):
        """Generator returning all 'wrong' outaux spins corresponding to a given input. Lets you simulate the addition of an auxiliary spin of value tempaux.

        If a feasible auxiliary array has been set, then both (correct_out, correct_aux) AND (correct_out, wrong_aux) are considered correct, as both contain the correct output. Hence neither is returned.
        """

        numaux = self.A + tempaux.dim()
        iterator = (
            Spinspace(shape=(self.M,))
            if numaux == 0
            else Spinspace(shape=(self.M, numaux))
        )
        for outaux in iterator:
            # outaux either has shape (M) or (M, A') depending on whether or not
            # a feasible auxiliary array has been set
            if numaux == 0:
                out = outaux
            else:
                out, _ = outaux.split()

            # we only check the output component
            if out == self.fout(inspin):
                continue

            yield outaux

    ############################################
    ### VECTOR METHODS
    ############################################

    @cache
    def lvec(self, inspin, tempaux=Spin(0, shape=(0,))):
        """Returns the sign of the average normal vector for all constraints in the given input level"""
        s = inspin

        # Spinspace gets weird about empty shapes, this avoids it
        if tempaux.dim() > 0:
            correct_inout = Spin.catspin(self.inout(s).split() + (tempaux,))
        else:
            correct_inout = self.inout(s)

        # store the correct vspin
        correct_vspin = correct_inout.vspin().spin()

        # initialize the lvector as a numpy array of zeros
        tempG = self.G + tempaux.dim()
        lvec = np.zeros(int(tempG * (tempG + 1) / 2))
        for t in self.allwrong(inspin, tempaux):
            inout = Spin.catspin((s, t))
            diff = inout.vspin().spin() - correct_vspin
            lvec += diff / np.linalg.norm(diff)

        return lvec

    @cache
    def poslvec(self, inspin):
        """Returns the lvec of inspin with +1 added as a temporary auxiliary spin"""
        return self.lvec(inspin, tempaux=Spin(1, (1,)))

    @cache
    def neglvec(self, inspin):
        """Returns the lvec of inspin with -1 added as a temporary auxiliary spin"""
        return self.lvec(inspin, tempaux=Spin(0, (1,)))

    #############################################
    ### SOLVER METHODS
    #############################################

    def build_solver(self, input_spins=[]):
        """Builds the lp solver for this circuit, returns solver. Made purposefully verbose/explicit to aid in debugging, should be shortened eventually however."""

        G = self.G
        if input_spins == []:
            input_spins = [s for s in self.inspace]

        solver = pywraplp.Solver.CreateSolver("GLOP")
        inf = solver.infinity()

        # set all the variables
        params = {}
        for i in range(G):
            params[i, i] = solver.NumVar(-inf, inf, f"h_{i}")
            for j in range(i + 1, G):
                params[i, j] = solver.NumVar(-inf, inf, f"J_{i},{j}")

        # we treat case with and without aux separately
        for inspin in input_spins:
            correct_inout_pair = self.inout(inspin)
            for wrong in self.allwrong(inspin):
                inout_pair = Spin.catspin(spins=(inspin, wrong))

                # build the constraint corresponding the difference of correct and incorrect output
                constraint = solver.Constraint(0.001, inf)
                s = correct_inout_pair.spin()
                t = inout_pair.spin()
                for i in range(G):
                    constraint.SetCoefficient(params[i, i], float(t[i] - s[i]))
                    for j in range(i + 1, G):
                        constraint.SetCoefficient(
                            params[i, j], float(t[i] * t[j] - s[i] * s[j])
                        )

        # print(f"skipped {tally}")
        return solver

    #############################################
    ### CHECK SPECIFIC hJ_vecs ON INPUT LEVELS
    #############################################

    def energy(self, spin: Spin, hvec: np.ndarray):
        return np.dot(spin.vspin().spin(), hvec)

    def level(self, hvec, inspin, weak=False):
        """Checks whether a given input level is satisfied by hvec

        Parameters
        ----------
            hvec : np.ndarray
                the hamiltonian vector/hJ_vector to test
            inspin : Spin
                the input to examine
            weak : bool (default=False)
                check for strong satisfaction of the constraints ( correct < incorrect ) or weak satisfaction ( correct <= incorrect )

        Returns: (bool) True if input level satisfied by hvec, False otherwise
        """

        # bool describing whether this level is satisfied by ham_vec
        satisfied = True

        # store info about the correct in/out pair
        correct_energy = self.energy(self.inout(inspin), hvec)

        for outspin in self.allwrong(inspin):
            energy = self.energy(Spin.catspin((inspin, outspin)), hvec)

            if weak == False and energy <= correct_energy:
                return False
            elif weak == True and energy < correct_energy:
                return False
        return True

    def passlist(self, hvec: tuple[float], weak=False):
        """Returns a list with one entry per input with 1 if the input level is satsified and 0 otherwise.

        Parameters
        ----------
            hvec : np.ndarray
                the hamiltonian vector/hJ_vector to test
            inspin : Spin
                the input to examine
            weak : bool (default=False)
                check for strong satisfaction of the constraints ( correct < incorrect ) or weak satisfaction ( correct <= incorrect )

        Returns
        -------
            passlist : list[int]
                one entry per input spin; 1 if input level satisfied, 0 otherwise
        """

        return [int(self.level(hvec, inspin, weak)) for inspin in self.inspace]

    def levels(self, hvec, inspins: list[Spin] = [], weak=False, list_fails=False):
        """Checks whether a list of spins are satisfied by hvec.

        Parameters
        ----------
            hvec : np.ndarray
                the hamiltonian vector/hJ_vector to test
            inspins : list[Spin] (default=[])
                the list of inputs to examine. If no list provided, then checks all inputs.
            weak : bool (default=False)
                check for strong satisfaction of the constraints ( correct < incorrect ) or weak satisfaction ( correct <= incorrect )

        Returns: (bool) True if input levels all satisfied by hvec, False otherwise

        """
        if inspins == []:
            inspins = self.inspace.tospinlist()

        for inspin in inspins:
            condition = self.level(hvec, inspin, weak=weak)
            if condition == False:
                return False

        return True


class IMul(PICircuit):
    """Ising Multiply Circuit"""

    def __init__(self, N1: int, N2: int, A: int = 0):
        super().__init__(N=N1 + N2, M=N1 + N2, A=A)
        self.inspace = Spinspace(shape=(N1, N2))
        self.N1 = N1
        self.N2 = N2

    def fout(self, inspin: Spin):
        # get the numbers corresponding to the input spin
        num1, num2 = inspin.splitint()

        # multiply spins as integers and convert into spin format
        result = Spin(spin=num1 * num2, shape=(self.M,))
        return result


class IBFunc(PICircuit):
    """An 'Ising Boolean Function', i.e. a PICircuit with only one output"""

    def __init__(self, N, fout: Callable):
        super().__init__(N=N, M=1, A=0)
        self._fout = fout

    def fout(self, inspin):
        return self._fout(inspin)

    def wfout(self, inspin):
        return self._fout(inspin).inv()

    def wf(self, inspin: Spin | int) -> Spin:
        # ensure is of type Spin
        if isinstance(inspin, int):
            inspin = self.inspace.getspin(inspin)

        out = self.wfout(inspin)
        aux = self.faux(inspin)
        if aux is None:
            return out
        else:
            return Spin(spin=(out.asint(), aux.asint()), shape=(self.M, aux.dim()))

    def winout(self, inspin):
        return Spin.catspin((inspin, self.wf(inspin)))

    def all_true_inputs(self):
        return [inspin for inspin in self.inspace if self.fout(inspin).asint() == 1]

    def all_false_inputs(self):
        return [inspin for inspin in self.inspace if self.fout(inspin).asint() == 0]

    def spin_action(self, spin):
        def new_fout(inspin):
            res = np.multiply(self.inout(inspin).spin(), spin.spin())
            return Spin(spin=res, shape=spin.shape)

        return IBFunc(self.N, fout=new_fout)

    @staticmethod
    def bool2ising(boolfunc):
        """Creates a function with codomain {Spin(1), Spin(-1)} from a function with codomain {True, False}"""

        def fout(inspin):
            return (
                Spin(spin=1, shape=(1,))
                if boolfunc(inspin)
                else Spin(spin=0, shape=(1,))
            )

        return fout

    def __add__(self, other):
        if not (isinstance(other, IBFunc) or isinstance(other, IMatch)):
            raise TypeError(f"Cannot add type (IBFunc) with type {type(other)}")

        def newfout(inspin):
            out1 = bool(self.fout(inspin).asint())
            out2 = bool(other.fout(inspin).asint())
            return out1 or out2

        return IBFunc(N=self.N, fout=IBFunc.bool2ising(newfout))

    @staticmethod
    def aux_and_hvec(circuit1, hvec1, matchcircuit):
        import math

        # difference between energy of (original) right and wrong
        def D(s):
            return circuit1.energy(
                spin=circuit1.inout(s), hvec=hvec1
            ) - circuit1.energy(spin=circuit1.winout(s), hvec=hvec1)

        def newaux(val: int):
            return Spin(spin=int(val == 1), shape=(1,))

        # store the graph size of circuit1
        G = circuit1.G

        # store the matchspin as snew
        snew = matchcircuit.matchspin

        # create an aux dictionary
        aux_dict = {}

        # the indices of the out and aux in the J matrix
        out = circuit1.N
        aux = circuit1.N + 1

        # store the original h and J of circuit1
        h1 = hvec1[: circuit1.G]
        J1 = hvec1[circuit1.G :]

        # set new h's and J's
        hnew = np.concatenate((h1, np.array([0])))
        Jnew = np.zeros((G + 1, G + 1))
        Jnew[np.triu_indices(Jnew.shape[0] - 1, k=1)] = J1

        # set the J_out,aux term and the corresponding auxiliary
        Jnew[out, aux] = -math.ceil(D(snew) / 2 + 1)
        aux_dict[snew] = newaux(1)


class IMatch(IBFunc):
    """PICircuit whose logic is a function f:S^N --> {-1,1} is f(s) = 1 if s = spin and f(s) = 0 otherwise"""

    def __init__(self, N, matchspin):
        if isinstance(matchspin, Spin):
            self.matchspin = matchspin
        else:
            self.matchspin = Spin(matchspin, shape=(N,))

        def matcher(inspin):
            return Spin(spin=int(inspin == self.matchspin), shape=(1,))

        super().__init__(N=N, fout=matcher)

    def hvec(self):
        """Generates a valid h,J vector for this circuit. Takes shape
        vec = (h_1,...,h_G, J_{1,2},...,J_{1,G},J_{2,3}...,J_{G-1,G})
        """
        correct_inout = self.inout(self.matchspin)
        incorrect_inout = Spin.catspin(
            (self.matchspin, self.fout(self.matchspin).inv())
        )
        h = np.array([0 for _ in range(self.N)] + [2 * self.N - 1])
        J = -2 * correct_inout.pairspin().spin()
        return np.concatenate((h, J))


class AND(IMatch):
    """Ising Multiply Circuit"""

    def __init__(self):
        super().__init__(N=2, matchspin=Spin(spin=1, shape=(1,)))


#####################################
### Methods to run as tests
#####################################


def example():
    G = IMul(2, 2)

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
