from spinspace import Spinspace, Spin
from abc import abstractmethod
from ortools.linear_solver import pywraplp
import spinspace as ss
import numpy as np


class PICircuit:
    """Base class for Pre Ising Circuits"""

    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M
        self.A = 0
        self.inspace = Spinspace(tuple([self.N]))
        self.outspace = Spinspace(tuple([self.M]))
        self.auxspace = None
        self.spinspace = Spinspace((self.N, self.M, self.A))
        self._graph = None
        self._aux_array = []

    #############################################
    # Properties
    #############################################
    @property
    def G(self):
        return self.N + self.M + self.A

    @property
    def hJ_num(self):
        return int(self.G*(self.G + 1)/2)

    @property
    def graph(self):
        if self._graph == None:
            self._graph = self.generate_graph()
        return self._graph

    @property
    def fulloutspace(self):
        if self.A == 0 or self.A == None:
            return self.outspace
        else:
            return self._outauxspace

    def set_aux(self, aux_array: list[list]):
        """
        Sets the auxiliary array. Expects a list of lists. Intelligently converts provided input into a list of Spins.

        Parameters
        ----------
        aux_array : list[list]
            the auxiliary array to use for this PICircuit. Must be of consistent shape, either (2^N, A) or (A, 2^N) (in numpy notation).
        """
        # reinitialize _aux_array
        self._aux_array = []

        # convert to numpy array
        aux_array = np.array(aux_array)

        # check to ensure either rows or columns matches 2^N
        if aux_array.shape[0] != 2 ** self.N and aux_array.shape[1] != 2**self.N:
            raise ValueError("The aux_array must have one auxiliary state per input!")

        # we want the first index to correspond to the input, not to the coordinate of the aux state
        if aux_array.shape[1] == 2**self.N:
            aux_array = aux_array.T

        self.A = len(aux_array[0])
        self.auxspace = Spinspace(shape=(self.A,))
        self._outauxspace = Spinspace(shape=(self.M, self.A))

        # check for consistent length and store aux_array as list of Spins
        for i in range(aux_array.shape[0]):
            row = aux_array[i]
            if len(row) != self.A:
                raise ValueError("Not all auxiliary states are the same length!")

            self._aux_array.append(Spin(spin=row, shape=(self.A,)))

    def energy(self, spin: Spin, ham_vec: np.ndarray):
        return np.dot(spin.vspin().spin(), ham_vec)

    @abstractmethod
    def fout(self, inspin: Spin) -> Spin:
        pass

    def faux(self, inspin: Spin) -> None | Spin:
        if self.A == 0:
            return None
        else:
            return self._aux_array[inspin.asint()]

    def f(self, inspin: Spin | int) -> Spin:
        # ensure is of type Spin
        if isinstance(inspin, int):
            inspin = self.inspace.getspin(inspin)

        out = self.fout(inspin)
        aux = self.faux(inspin)
        if aux is None:
            return out
        else:
            return Spin.catspin(spins=(out, aux))

    def inout(self, inspin: Spin | list[Spin]):
        """Returns the (in, out) pair corresponding to (s,f(s)) for an input spin s. If a list of Spins is provided instead then a list of (in, out) pairs returned."""
        if isinstance(inspin, Spin):
            return Spin.catspin((inspin, self.f(inspin)))
        else:
            return [Spin.catspin((s, self.f(s))) for s in inspin]

    def generate_graph(self):
        graph = [self.inout(s) for s in self.inspace]
        return graph

    def lvec(self, inspin):
        """Returns the sign of the average normal vector for all constraints in the given input level"""
        s = inspin
        normal = np.zeros(int(self.G * (self.G + 1) / 2))
        for t in self.outspace:
            inout = Spin.catspin((s, t))
            normal += inout.vspin().spin()

        return np.sign(normal - self.M * self.inout(s).vspin().spin())

    #############################################
    # Solver
    #############################################

    def init_solver(self):
        """Initialize the linear solver from ortools"""
        pass

    # old level checkers
    def level(self, inspin, ham_vec, weak=False, more_info=False, print_energies=False):
        """Returns information about the level

        Parameters
        ----------
        inspin : Spin
            an input spin
        weak : bool
            check for weak satisfaction of constraints (<=) rather than satisfaction (<)

        """

        # bool describing whether this level is satisfied by ham_vec
        satisfied = True

        # store info about the correct in/out pair
        s = inspin
        correct_int = self.f(inspin).asint()
        correct_spin = self.inout(s)
        correct_key = correct_spin.asint()
        correct_energy = self.energy(correct_spin, ham_vec)

        # dictionary to store hamiltonian values
        ham = {correct_key: correct_energy}

        if print_energies:
            print(f"Correct output {correct_int} had energy {correct_energy}")

        # iterate through the level
        for i in range(self.fulloutspace.size):
            t = self.fulloutspace.getspin(i)
            if t.asint() == correct_int:
                continue

            spin = Spin.catspin((s, t))
            energy = self.energy(spin=spin, ham_vec=ham_vec)
            if print_energies:
                print(f"Output {t.asint()} had energy {energy}")

            # satisfied
            if weak == False:
                if energy <= correct_energy:
                    satisfied = False
                    if more_info == False:
                        return satisfied
            else:
                if energy < correct_energy:
                    satisfied = False
                    if more_info == False:
                        break

            ham[spin.asint()] = energy

        if more_info:
            return satisfied, dict(sorted(ham.items(), key=lambda x: x[1]))
        else:
            return satisfied

    def levels(
        self,
        inspins: list[Spin],
        ham_vec,
        list_fails=False,
        weak=False,
        print_energies=False,
        flag=False,
    ):
        """Returns information about the levels of a list of spins. By default, returns True if all levels are satisfied by ham_vec, False otherwise. If list_fails = True, then it returns a list of the input spins whose levels are not satisfied by ham_vec"""
        fails = []
        for s in inspins:
            condition = self.level(
                inspin=s, ham_vec=ham_vec, weak=weak, print_energies=print_energies
            )
            if condition == False:
                if list_fails == False:
                    return False
                fails.append(s)

        if list_fails:
            return fails

        return True

    def build_solver(self, input_spins=[]):
        """Builds the lp solver for this circuit, returns solver"""

        if input_spins == []:
            input_spins = [s for s in self.inspace]

        solver = pywraplp.Solver.CreateSolver("GLOP")
        inf = solver.infinity()

        # set all the variables
        params = {}
        for i in range(self.G):
            params[i,i] = solver.NumVar(-inf, inf, f'h_{i}')
            for j in range(i+1, self.G):
                params[i,j] = solver.NumVar(-inf,inf, f'J_{i},{j}')

        # add the constraints without iterating through spinspace
        tally = 0
        constraints = []
        for inspin in self.inspace:
            correct_out = self.fout(inspin)
            correct_aux = self.faux(inspin)
            correct_inout_pair = self.inout(inspin)
            for outspin in self.fulloutspace:
                inout_pair = Spin.catspin(spins=(inspin, outspin)) 
                out, aux = outspin.splitint()
                
                if inout_pair == correct_inout_pair:
                    tally += 1
                    continue
                elif out == correct_out.asint() and aux != correct_aux.asint():
                    tally += 1
                    continue

                # build the constraint corresponding the difference of correct and incorrect output
                constraints.append(solver.Constraint(0.001, inf))
                s = correct_inout_pair.spin()
                t = inout_pair.spin()
                for i in range(self.G):
                    constraints[-1].SetCoefficient(params[i,i], float(t[i] - s[i]))
                    for j in range(i+1, self.G):
                        constraints[-1].SetCoefficient(params[i,j], float(t[i]*t[j] - s[i]*s[j]))
                

        #print(f"skipped {tally}")
        return solver


class IMul(PICircuit):
    """Ising Multiply Circuit"""

    def __init__(self, N1: int, N2: int):
        super().__init__(N=N1 + N1, M=N1 + N2)
        self.inspace = Spinspace(shape=(N1, N2))
        self.N1 = N1
        self.N2 = N2
        self.N = N1 + N2

    def fout(self, inspin: Spin):
        # get the numbers corresponding to the input spin
        num1, num2 = inspin.splitint()

        # multiply spins as integers and convert into spin format
        result = Spin(spin=num1 * num2, shape=(self.M,))
        return result


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

#####################################
### Methods to run as tests
#####################################

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
