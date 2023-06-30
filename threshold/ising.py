from functools import cache
from copy import Error
from spinspace import Spinspace, Spin
from abc import abstractmethod
from ortools.linear_solver import pywraplp
import spinspace as ss
import numpy as np
import math


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

    def __init__(self, N: int, M: int, A: int = 0, auxlist=None):
        self.N = N
        self.M = M
        self.A = A
        self.inspace = Spinspace((self.N,))
        self.outspace = Spinspace((self.M,))
        self.auxspace = Spinspace((self.A,))
        self.spinspace = Spinspace((self.N, self.M, self.A))
        self._graph = None
        self._aux_dict = {}

        if auxlist:
            self.set_all_aux(auxlist)

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

    @property
    def outauxspace(self):
        if self.A:
            return self._outauxspace
        else:
            return self.outspace

    def Gin(self, inspin):
        return self.N + self.M + self.Ain(inspin)

    #################################
    ### Private methods
    #################################

    def _generate_graph(self):
        graph = [self.inout(s) for s in self.inspace]
        return graph

    #################################
    ### AUXILIARY STATE METHODS
    #################################

    def Ain(self, inspin):
        """Returns the number of auxiliary spins set for the provided input"""
        try:
            return self._aux_dict[inspin].dim()
        except KeyError:
            return 0

    def add_single_aux(self, inspin: Spin, auxval: int):
        """Adds a single aux spin (either +1 or -1) to the specified input"""
        try:
            self._aux_dict[inspin] = Spin.append(self._aux_dict[inspin], auxval)
        except KeyError:
            self._aux_dict[inspin] = Spin(spin=int((auxval + 1) / 2), shape=(1,))

    def set_aux(self, inspin: Spin, auxspin):
        """Associates an auxiliary spin to the specified input"""
        self._aux_dict[inspin] = auxspin

    def set_all_aux(self, aux_array):
        """
        Sets the auxiliary array. Expects a list of lists. Intelligently converts provided input into a list of Spins.

        Parameters
        ----------
        aux_array : list[list]
            the auxiliary array to use for this PICircuit. Must be of consistent shape, either (2^N, A) or (A, 2^N) (in numpy notation).
        """

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

        # update self.auxspace if necessary
        self.auxspace = Spinspace(shape=(self.A,))

        # check for consistent length and store aux_array as list of Spins
        for i in range(aux_array.shape[0]):
            row = aux_array[i]
            if len(row) != self.A:
                raise ValueError("Not all auxiliary states are the same length!")

            self._aux_dict[self.inspace[i]] = Spin(spin=row, shape=(self.A,))

    def get_aux_array(self, binary=False):
        """Returns the aux array of this circuit"""
        if self.A == 0:
            return None
        if binary:
            aux_array = [
                self._aux_dict[inspin].binary().tolist() for inspin in self.inspace
            ]
        else:
            aux_array = [
                self._aux_dict[inspin].spin().tolist() for inspin in self.inspace
            ]
        return np.array(list(zip(*aux_array)))

    @staticmethod
    def gen_random_auxarray(N, A, binary=False):
        auxspace = Spinspace((A,))
        if binary:
            auxarray = [list(auxspace.rand().binary()) for _ in range(2**N)]
        else:
            auxarray = [list(auxspace.rand().spin()) for _ in range(2**N)]

        return list(zip(*auxarray))

    def check_aux_all_set(self, inspins=[]):
        if inspins == []:
            inspins = self.inspace

        first = self.Ain(list(inspins)[0])
        return all(first == self.Ain(x) for x in inspins)

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
            if isinstance(inspin, int):
                inspin = self.inspace[inspin]
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
        outauxspace = Spinspace(shape=(self.M, self.Ain(inspin)))
        for outaux in outauxspace:
            yield Spin.catspin((inspin, outaux))

    def allwrongout(self, inspin):
        iterator = Spinspace(shape=(self.M,))
        for out in iterator:
            # we only check the output component
            if out == self.fout(inspin):
                continue

            yield out

    def allwrong(self, inspin, tempaux=Spin(0, shape=(0,))):
        """Generator returning all 'wrong' outaux spins corresponding to a given input. Lets you simulate the addition of an auxiliary spin of value tempaux.

        If a feasible auxiliary array has been set, then both (correct_out, correct_aux) AND (correct_out, wrong_aux) are considered correct, as both contain the correct output. Hence neither is returned.
        """

        numaux = self.Ain(inspin) + tempaux.dim()
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
        tempG = self.Gin(inspin) + tempaux.dim()
        lvec = np.zeros(int(tempG * (tempG + 1) / 2))
        for t in self.allwrong(inspin, tempaux):
            inout = Spin.catspin((s, t))
            diff = inout.vspin().spin() - correct_vspin
            lvec += diff / np.linalg.norm(diff)

        return lvec

    @cache
    def lvec_noaux(self, inspin):
        """Returns the sign of the average normal vector for all constraints in the given input level, ignoring the auxiliary spins"""
        s = inspin
        correct_inout = Spin.catspin((s, self.fout(s)))

        # store the correct vspin
        correct_vspin = correct_inout.vspin().spin()

        # initialize the lvector as a numpy array of zeros
        tempG = self.N + self.M
        lvec = np.zeros(int(tempG * (tempG + 1) / 2))
        for t in self.allwrongout(s):
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

    @cache
    def avglvec_dist(self, inspin):
        diffs = [
            np.linalg.norm(self.lvec(inspin) - self.lvec(s))
            for s in self.inspace.copy()
        ]
        return sum(diffs) / self.N

    @cache
    def fastavglvec_dist(self, inspin):
        diffs = [
            np.linalg.norm(self.lvec_noaux(inspin) - self.lvec_noaux(s))
            for s in self.inspace.copy()
        ]
        return sum(diffs) / self.N

    #############################################
    ### SOLVER METHODS
    #############################################

    def build_solver(self, input_spins=[]):
        """Builds the lp solver for this circuit, returns solver. Made purposefully verbose/explicit to aid in debugging, should be shortened eventually however."""

        # check that each input has same number of auxiliary states
        if self.check_aux_all_set(input_spins) == False:
            raise Error(
                f"Not all auxiliary states are the same size! Cannot build constraints.\n { {s : self.Ain(s) for s in input_spins} }"
            )

        G = self.Gin(self.inspace[0])
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

    @cache
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

    @cache
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

    def __init__(self, N1: int, N2: int, A: int = 0, auxlist=None):
        super().__init__(N=N1 + N2, M=N1 + N2, A=A, auxlist=auxlist)
        self.inspace = Spinspace(shape=(N1, N2))
        self.N1 = N1
        self.N2 = N2

    def fout(self, inspin: Spin):
        if isinstance(inspin, int):
            inspin = self.inspace[inspin]
        # get the numbers corresponding to the input spin
        num1, num2 = inspin.splitint()

        # multiply spins as integers and convert into spin format
        result = Spin(spin=num1 * num2, shape=(self.M,))
        return result

    @staticmethod
    def gen_random_circuit(N1, N2, A):
        circuit = IMul(N1, N2)
        circuit.set_all_aux(PICircuit.gen_random_auxarray(N1 + N2, A))
        return circuit


class AND(PICircuit):
    """Ising Multiply Circuit"""

    def __init__(self):
        super().__init__(N=2, M=1)
        self.inspace = Spinspace(shape=(1, 1))

    def fout(self, inspin: Spin):
        # get the numbers corresponding to the input spin
        num1, num2 = inspin.splitint()

        output = 1 if (num1 == 1 and num2 == 1) else 0
        # multiply spins as integers and convert into spin format
        result = Spin(spin=output, shape=(self.M,))
        return result


class BCircuit(PICircuit):
    """Class for defining a boolean circuit"""

    def __init__(self, N, func):
        super().__init__(N=N, M=1)
        if callable(func):
            self.vals = [func(k) for k in range(1 << N)]
        else:
            self.vals = func

        assert len(self.vals) == 1 << N

    def __call__(self, a):
        return self.vals[a]

    def fout(self, inspin):
        out = self.vals[inspin.asint()]
        out = out if out else 0
        return Spin(out, shape=(1,))

    def func(self):
        def f(a):
            return self.vals[a]

        return f

    def asdict(self):
        return {Spin(a, (self.N,)): self.vals[a] for a in range(1 << self.N)}

    def dual(self):
        vals = [not self.vals[~a] for a in range(1 << self.N)]
        return BCircuit(self.N, vals)

    def selfdual(self):
        dvals = [int(not self.vals[~a]) for a in range(1 << self.N)]
        return BCircuit(self.N + 1, self.vals + dvals)

    def neutralizable_solver(self):
        """Builds the lp solver for this circuit, returns solver. Made purposefully verbose/explicit to aid in debugging, should be shortened eventually however."""

        G = self.G
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
            s = correct_inout_pair.spin()
            for wrong in self.allwrong(inspin):
                inout_pair = Spin.catspin(spins=(inspin, wrong))

                # build the constraint corresponding the difference of correct and incorrect output
                constraint = solver.Constraint(1, inf)
                t = inout_pair.spin()
                for i in range(G):
                    constraint.SetCoefficient(params[i, i], float(t[i] - s[i]))
                    for j in range(i + 1, G):
                        constraint.SetCoefficient(
                            params[i, j], float(t[i] * t[j] - s[i] * s[j])
                        )
            nextin = inspin.asint() + 1
            if nextin < 1 << self.N:
                t = self.inout(self.inspace[nextin])
                neut_constraint = solver.Constraint(0, 0)
                for i in range(G):
                    neut_constraint.SetCoefficient(params[i, i], float(t[i] - s[i]))
                    for j in range(i + 1, G):
                        neut_constraint.SetCoefficient(
                            params[i, j], float(t[i] * t[j] - s[i] * s[j])
                        )

        # print(f"skipped {tally}")
        return solver


def check_neutralizability():
    andcirc = BCircuit(2, [0, 0, 0, 1])
    xorcirc = BCircuit(2, [0, 1, 1, 0])
    and3circ = BCircuit(3, [0, 0, 0, 0, 0, 0, 0, 1])
    andselfdual = andcirc.selfdual()

    solver1 = andcirc.neutralizable_solver()
    solver2 = xorcirc.neutralizable_solver()
    solver3 = and3circ.neutralizable_solver()
    solver4 = andselfdual.neutralizable_solver()

    for a in range(8):
        print(a, and3circ(a))

    print(f"AND between 2: {solver1.Solve()}")
    print(f"          XOR: {solver2.Solve()}")
    print(f"AND between 3: {solver3.Solve()}")
    print(f" AND selfdual: {solver4.Solve()}")


if __name__ == "__main__":
    check_neutralizability()
