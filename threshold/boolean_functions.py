import math


#########################################
## METHODS OF THE BOOLEAN FUNCTION MODULE
#########################################


def dec2bin(num, fill):
    return list(int(a) for a in bin(num)[2:].zfill(fill))


def bin2dec(blist):
    N = len(blist)
    return sum([1 << (N - i - 1) for i in range(N) if blist[i]])


def bin2hex(b):
    return "".join(
        [
            format(int("".join([str(int(x)) for x in b[i : (i + 4)]]), 2), "x")
            for i in range(0, len(b), 4)
        ]
    )


def hex2bin(num, fill):
    return dec2bin(int(num, 16), fill)


def dual(func):
    N = int(math.log2(len(func)))
    return [int(not func[~a]) for a in range(1 << N)]


def selfdual(func):
    N = int(math.log2(len(func)))
    func = list(func)
    dvals = [int(not func[~a]) for a in range(1 << N)]
    return func + dvals


#########################################
## CLASSES IN THE BOOLEAN FUNCTION MODULE
#########################################


class BooleanFunction:
    def __init__(self, num, input_size):
        self.id = num
        self.N = input_size
        self.vals = dec2bin(num, 1 << self.N)

    def __call__(self, index):
        return self.vals[index]

    def __getitem__(self, index):
        return self.vals[index]

    def __add__(self, other):
        """Two different behaviors depending on whether other is an integer (in which case this is a spin action) or another boolean function (in which case this is pointwise add)"""
        if isinstance(other, int):
            # separate other into an integer representing the input and the output portion
            # of the binary string
            a = int(bin(other)[2:].zfill(self.N + 1)[:-1], 2)
            b = other % 2
            vals = [(self.vals[k ^ a] + b) % 2 for k in range(1 << self.N)]
            return BooleanFunction(num=bin2dec(vals), input_size=self.N)
        else:
            if self.N != other.N:
                raise IndexError("Cannot add functions of different lengths!")
            vals = [(self.vals[i] + other.vals[i]) % 2 for i in range(1 << self.N)]
            return BooleanFunction(num=bin2dec(vals), input_size=self.N)

    def __eq__(self, other):
        return other.vals == self.vals

    def __hash__(self):
        return hash(tuple([self.id, self.N]))

    def func(self):
        """Returns a callable representing this function"""

        def f(num):
            return self.vals[num]

        return f

    def get_vals(self):
        """Getter for the values of this function"""
        return self.vals

    def dual(self):
        return [int(not self.vals[~a]) for a in range(1 << self.N)]

    def selfdual(self):
        dvals = self.dual()
        return self.vals + dvals

    def mapping_table(self):
        return [
            f"{bin(k)[2:].zfill(self.N)} ~~> {self.vals[k]}" for k in range(1 << self.N)
        ]


class SpinOrbit:
    def __init__(self, boolfunc, input_size=-1):
        # allow for constructor to take same constructor as BooleanFunction
        if isinstance(boolfunc, int):
            assert input_size != -1
            boolfunc = BooleanFunction(boolfunc, input_size)

        self.N = boolfunc.N
        self.orbit = set([boolfunc + k for k in range(1 << (self.N + 1))])
        self.size = len(self.orbit)

    def __hash__(self):
        return hash(tuple(bf.id for bf in self.orbit) + (self.N,))

    def orbit_asint(self):
        return set([bf.id for bf in self.orbit])


def display_spin_action(func, spin):
    table = [func.mapping_table()[k] + "   " for k in range(1 << func.N)]
    table = zip(table, (func + spin).mapping_table())
    for row in table:
        print("".join(row))


def get_orbits(N):
    orbits = set([])
    bf_to_check = set(range(1 << (1 << N)))
    while len(bf_to_check) > 0:
        elem = list(bf_to_check)[0]
        func = BooleanFunction(num=elem, input_size=N)
        orbit = SpinOrbit(func)
        orbits.add(orbit)
        print(bf_to_check)
        print(orbit.orbit_asint())
        bf_to_check = bf_to_check.difference(set(orbit.orbit_asint()))
        print(f"{len(bf_to_check)} remaining")

    return orbits


def check_strong_neutralizability_among_orbits(N):
    from classify_boolean_functions import (
        check_strong_neutralizability as sncheck,
        dual,
        selfdual,
    )

    orbits = list(get_orbits(N))
    success_rate = []
    trigger = False
    for i, orbit in enumerate(orbits):
        successes = 0
        for bf in orbit.orbit:
            successes += 1 if sncheck(func=bf.vals) else 0

        success_rate.append(successes / orbit.size)
        print(f"Orbit #{i} has success rate {success_rate[-1]}")

        if success_rate[-1] != 0 and success_rate[-1] != 1:
            trigger = True

    data = list(zip(orbits, success_rate))
    good_orbits = [row[0] for row in data if row[1] == 1]
    print(f" {len([row for row in data if row[1] == 1])} strongly neutralizable orbits")
    for k in range(1, N + 2):
        count = len([row for row in data if len(row[0].orbit) <= 1 << k])
        print(f"  {count} orbits of size {1<< k} or less")

    for i, orbit in enumerate(good_orbits):
        dualorb = [dual(bf.vals) for bf in orbit.orbit]
        selfdualorb = [selfdual(bf.vals) for bf in orbit.orbit]
        hexorb = [bin2hex(bf.vals) for bf in orbit.orbit]
        print(f"GOOD ORBIT {i}")
        print(f"  {hexorb}")
        print(f"  {[bin2hex(bf) for bf in dualorb]}")
        print(f"  {[bin2hex(bf) for bf in selfdualorb]}")


def save_strongly_neutralizable_orbits(N):
    from classify_boolean_functions import check_strong_neutralizability as sncheck

    orbits = get_orbits(N)
    good_orbits = []
    for orbit in orbits:
        bf = list(orbit.orbit)[0]
        if sncheck(func=bf.vals):
            good_orbits.append([bin2hex(b=b.vals) for b in orbit.orbit])

    with open(f"data/SN_spinorbits_boolfunc_dim={N}.dat", "w") as file:
        for row in good_orbits:
            file.write(str(row) + "\n")


def display_orbits(N):
    orbits = get_orbits(N)
    print(f"ORBIT DECOMPOSITION OF BOOLEAN_FUNCTIONS ON {N} INPUTS")
    print(f"  # orbits: {len(orbits)}")
    print(
        f"  size of orbits: {','.join([str(len(orbit.orbit_asint())) for orbit in orbits])}"
    )
    print(
        f"  sum of orbit sizes: {sum([len(orbit.orbit_asint()) for orbit in orbits])}"
    )
    print(f"Expected: {1 << (1 << N)}")


if __name__ == "__main__":
    N = 3
    spin = [0, 0, 0, 1, 1]
    """
    print(spin, bin2dec(spin))
    func = BooleanFunction(36478, 4)
    display_spin_action(func, bin2dec(spin))
    """
    check_strong_neutralizability_among_orbits(N)
    # save_strongly_neutralizable_orbits(N)
