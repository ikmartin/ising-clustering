import argparse
import pickle

def inSpins(i, j):

    # The total number of input spins in the Ising system

    return i + j

################################################################################

def outSpins(op, i, j):

    # The total number of output spins in the Ising system

    if op == 'ADD':
        n = max(i, j) + 1;
    elif op == 'MUL' and min(i, j) > 1:
        n = i + j
    else:
        n = max(i, j)

    return n

################################################################################

def totSpins(op, i, j, k):

    # The total number of all the spins in the Ising system

    return inSpins(i, j) + outSpins(op, i, j) + k

################################################################################

def auxVariables(i, j, k):

    # The total number of auxiliary variables with value +/- 1

    m = inSpins(i, j)

    return k * ( 1 << m )

################################################################################

def psiVariables(op, i, j, k):

    # The total number of real variables which includes all of the
    # Hamiltonian field strengths (h) and interaction potentials (J)

    m, n, l = inSpins(i, j), outSpins(op, i, j), totSpins(op, i, j, k)

    return ( ( n + k ) * ( m + l + 1 ) ) >> 1

################################################################################

# Create the parser
my_parser = argparse.ArgumentParser(prog='make-ising',
                                    usage='%(prog)s [options]'
                                    description='Tool for creating and saving Ising graphs and circuits')

# Create a dictionary of recognized operations

opDict = { 'AND' : '&', 'OR' : '|', 'XOR' : '^',
           'ADD' : '+', 'MUL' : '*' }


################################################################################

def makeIsing(op, N1, N2, A):
    import ising

    if op == 'MUL':
        return ising.IMul(N1=N1,N2=N2,A=A)

    else:
        N, M, G = inSpins(N1, N2), outSpins(op, N1, N2), totSpins(op, N1, N2, k)
        return ising.PICircuit(N=N,M=M,A=A)

################################################################################

if __name__ == '__main__':

    # Parse the command line
    parser = ArgumentParser()
    parser.add_argument('xspins', type = int, help = 'Number of x-spins')
    parser.add_argument('yspins', type = int, help = 'Number of y-spins')
    parser.add_argument('aspins', type = int, help = 'Number of a-spins')
    parser.add_argument('--op', type = str, required = True,
        choices=opDict.keys(),
        help = 'Operation to be performed on the input values')
    parser.add_argument('--of', type = str, default = None,
        help = 'Name of output file to write to')
    arg = parser.parse_args()

    # Calculate parameters for this problem
    op, N1, N2, A = arg.op, arg.xspins, arg.yspins, arg.aspins
    G = makeIsing(op,N1,N2,A)
    
    file = arg.of
    if of != None:
        ofile = open(of + '.picirc','w')
        pickle.dump(G, ofile)

################################################################################
