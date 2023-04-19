import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import time
import distance as dist
import ising as ising
import spinspace

spin_spaces = {}

def dist_hist_single_spin(fi=0,
                          dname=None,
                          metric=None,
                          G=None,
                          num_hJ=None,
                          wait=False,
                          h=None,
                          J=None,
                          dirname='default',
                          filename=''):

    ''' Make a bar graph showing the distribution of ising distances. If no parameters are provided, will prompt user for input.

    Optional Params:
    *** fi (default=0): (int) index of the spin to measure against
    *** dname (default=None): (str) name of the metric used
    *** metric (default=None): (method) the metric to use
    *** G (default=None): dimension of the spin space
    *** num_hJ (default=None): a decimal integer converted to base 3 to derive h, J parameters, if applicable to metric
    *** wait (default=False): (bool) decide whether or not to wait and display h and J values prior to generating histogram
    '''

###### USER PROMPTS ######
    if dname == None: dname,metric = dist_select()

    # get spin space and cardinality of spin space
    if G == None: G = int(input("Enter spin space dimension: "))
    space_size = len(spinspace.get_spin_space(G))

    # get h,J values if applicable
    if h is None and J is None:
        if num_hJ == None: num_hJ = int(input(
                f"Enter a positive integer which will be converted into h and J values (0..{3**ising.num_params(N)}-1): "))
        h,J = ising.get_param_mod3(num_hJ, G)
 
    # display h and J values prior to histogram generation
    if wait: 
        print('Using the following values for h and J:')
        print(f'h = {h}')
        print(f'J = \n{J}')
        input('Press any key to continue...')

####### HISTOGRAM GENERATION #######
    fspin = spinspace.get_spin_space(G)[fi] # set the spin space and the fixed spin 

    # create array to store the distances
    d = []
    
    # status update
    print(f"Spin #{fi}: {fspin}")
    print("  Getting distribution... ", end='')
    # get all the distances
    for spin in spinspace.get_spin_space(G):
        d.append(metric(fspin,spin,h=h,J=J))
    
    print("done.")

    # create a histogram of the distances
    title = f"Distance distribution for spin {fspin}\nusing {dname} pseudometric"
    labels,counts = np.unique(np.array(d),return_counts=True)
    plt.bar(labels,counts,align='center',edgecolor='k')
    plt.title(title)
    plt.xticks()

    # save the figure
    if os.path.exists(f"dist-hist/{dname}/dim{G}/{dirname}") == False:
        os.makedirs(f"dist-hist/{dname}/dim{G}/{dirname}")
    plt.savefig(f"dist-hist/{dname}/dim{G}/{dirname}/{filename}.png")

    print(f"  Histogram saved to dist-hist/{dname}/dim{G}/{dirname}/{filename}.png")
    plt.close()

####################################################################

def dist_hist(dname=None,metric=None,G=None,num_hJ=None,wait=False):
    ''' Make a bar graph showing the distribution of ising distances

    Optional Params:
    *** N (default 4): (int) the dimension of the spin space
    *** fi (default 3): (int) the Fixed Index of the spin to measure against. It corresponds to the decimal representation of the spin.
    *** show (default False): (bool) display figure before saving
    '''

    if dname == None:
        dname,metric = dist_select()

    # get spin space and cardinality of spin space
    if G == None:
        G = int(input("Enter spin space dimension: "))
    space_size = len(spinspace.get_spin_space(G))

    # get h,J values if applicable
    if num_hJ == None:
        num_hJ = int(input(
            f"Enter a positive integer which will be converted into h and J values (0..{3**ising.num_params(G)}-1): "))
    h,J = ising.get_param_mod3(num_hJ, G)
    
    if wait:
        print('Using the following values for h and J:')
        print(f'h = {h}')
        print(f'J = {J}')
        input('Press any key to continue...')

    for i in range(0,space_size):
        dist_hist_single_spin(fi=i,
                              dname=dname,
                              metric=metric,
                              G=G,
                              num_hJ=num_hJ,
                              wait=False,
                              dirname=f'{num_hJ}',
                              filename=str(i))

####################################################################

def dist_hist_rand_param(G=None,
                         dname=None,
                         metric=None,
                         num0=None,
                         num1=None,
                         numJ0=None,
                         numJ1=None,
                         rounds=None,
                         ename=None):
    ''' Playing with random variations in h and J. Any arguments not passed are set via user prompt.

    Optional Params:
    *** G (default=None): (int) the size of the graph
    *** dname (default=None): (str) the name of the metric to use
    *** metric (default=None): (method) the metric to use
    *** num0 (default=None): (int) number of parameters (or h) which should be 0
    *** num1 (default=None): (int) number of parameters (or h) which should be 1
    *** numJ0 (default=None): (int) number of J parameters which should be 0
    *** numJ1 (default=None): (int) number of J parameters which should be 1
    *** rounds (default=None): (int) number of rounds to perform
    '''

    if dname is None: dname,metric = dist_select()

    # get spin space and cardinality of spin space
    if G is None: G = int(input("Enter spin space dimension: "))
    space_size = len(spinspace.get_spin_space(G))

    if numJ0 is None: 
        numJ0 = input("set number of 0's for J:")
        numJ1 = input("set number of 1's for J:")
        if numJ0: numJ0 = int(numJ0)
        else: numJ0 = None
        if numJ1: numJ1 = int(numJ1)
        else: numJ1 = None

    if num0 is None:
        num0 = int(input("set number of 0's for h:"))
        num1 = int(input("set number of 1's for h:"))

    if rounds is None:
        rounds = int(input("finally, how many rounds should this run? "))

    if ename is None:
        ename = input('FINALLY finally, what should this experiment be called?')
    for i in range(rounds):
        h,J,numhJ = ising.rand_param(G=G,
                               num0=num0,
                               num1=num1,
                               numJ0=numJ0,
                               numJ1=numJ1)

        dist_hist_single_spin(fi=int(2**G/2 - 2**G/4 + 2**G/8 + 3),
                              dname=dname,
                              metric=metric,
                              G=G,
                              wait=True if rounds < 20 else False,
                              h=h,
                              J=J,
                              dirname=ename,
                              filename=str(f'hJ_id={numhJ}'))


def dist_hist_menu():

    print('==========================\nDistance Histogram Submenu\n==========================')
    print('  1. Generate all histograms for all spins of fixed dimension')
    print('  2. Generate one histogram for varying parameterized distances')
    print('selection: ')
    select = int(input("\nselection: "))

    if select == 1:
        dist_hist()

    if select == 2:
        dist_hist_rand_param()


####################################################################
def make_dist_data_table():
    ''' Display the distances from the specified spin
    Params:
    *** N: (int) the dimension of the spin space to consider
    *** fi (default = 0): (int) the fixed index determining the spin to measure against
    '''

    # get metric to use
    dist1,metric1 = dist_select()
    
    # get spin space dimension
    N = int(input("Enter spin space dimension: "))

    # get the fixed index of the spin
    fi = int(input("Enter decimal representation of spin: "))

    # set the spin space
    space = ising.get_spin_space(N)

    print("\n=============================================================")
    print(f"Displaying {dist1} distance from {space[fi]}")
    print("=============================================================")

    table = []
    for spin in space:
        table.append([spin, metric1(space[fi],spin)])

    print(tabulate(table, headers=["2nd Spin", "Distance"]))

def makePICircuit():
    ''' Make a PICircuit CLI

    Returns: G: (PICircuit) the PICircuit instance specified by user
    '''
    print("Choose circuit type:")
    print("  1. MUL")
    print("  2. XOR")
    select = int(input("selection: "))
    if select == 1:
        # get MUL info
        N1 = int(input("  enter 1st input size:"))
        N2 = int(input("  enter 2nd input size:"))
        A = int(input("  enter number of auxiliary spins:"))
        
        # make ising graph
        G = ising.IMul(N1=N1,N2=N2,A=A)
    if select == 2:
        # get the necessary XOR info
        N = int(input("  enter input size:"))
        A = int(input("  enter number of auxiliary spins:"))
        
        # make ising graph
        G = ising.IXOR(N1=N,A=A)
        
    return G

def dist_select(msg=None):
    print("Select your distance:" if msg == None else msg)
    print("---------------------")
    i = 0
    for key in dist.dist_dict.keys():
        print(f"  {i}: {key}")
        i += 1
    
    print("ran dist_select")
    return dist.get_dist_func(int(input("selection: ")))

def dist_between_level():
    ''' Get info about levels of a preising circuit
    '''
    
    # get total size of spin space
    G = makePICircuit()
    N = G.N
    M = G.M + G.A

    if N not in spin_spaces.keys():
        spin_spaces[N] = ising.get_spin_space(N)

    if M not in spin_spaces.keys():
        spin_spaces[M] = ising.get_spin_space(M)

    # have user select a distance
    dname,metric = dist_select(msg="Finally, select a distance to use:") 

    levelNum = 0
    # loop through every input and 
    #   1. make a graph of the level and
    #   2. make a table of the level
    # make one final graph combining all existing graphs
    totx = range(2**(N+M))
    toty = [0 for a in totx]
    xcorrect = []
    
    # avg distance table made at the end
    circuitTable = []

    for ispin in spin_spaces[N]:

        # loop variables
        levelNum += 1
        levelTable = []
        # graphing variables
        x = range(2**M)
        y = [0 for a in x]
        
        # the correct output for the given input
        out = G.get_value(ispin)
        cspin = np.concatenate((ispin,out))

        # summed distance to correct value
        Dcorrect = 0
        numCorrect = 0 # should be 2**A
        Dwrong = 0
        numWrong = 0 # should be 2**(M+A) - 2**A

        for ospin in spin_spaces[M]:
            # fit ispin and ospin together
            spin = np.concatenate((ispin,ospin))
            isCorrect = G.isCorrect(ispin,ospin)
            
            # calculate distance
            d = metric(cspin,spin)

            if isCorrect:
                xcorrect.append(spinspace.spin2dec(spin))
                Dcorrect += d
                numCorrect += 1
            
            else:
                Dwrong += d
                numWrong += 1
            
            # append y values
            y[spinspace.spin2dec(ospin)] = d
            toty[spinspace.spin2dec(spin)] = d

            # append table value
            levelTable.append([ospin, d, isCorrect])

        print(f"Level #{levelNum}: input = {ispin}")
        print(tabulate(levelTable,headers=['output','distance from correct','Is Correct']))
        print(f'  Dwrong = {Dwrong}')
        print(f'  Dcorrect = {Dcorrect}')
        
        avgCor,avgWrong = (Dcorrect/numCorrect, Dwrong/numWrong)
        circuitTable.append([ispin,out[:G.M],out[G.M:],avgCor,avgWrong,'True' if avgCor < avgWrong else '.'])

    plt.plot(totx,toty,'-gD',markevery=xcorrect,mfc='black',mec='k')
    plt.title(f'Levels for {G.name}\n input = {ispin}')
    plt.show()

    if os.path.exists("levels-examine") == False:
        os.mkdir("levels-examine")
    if os.path.exists(f"levels-examine/{dname}") == False:
        os.mkdir(f"levels-examine/{dname}")
    if os.path.exists(f"levels-examine/{dname}/{G.name}") == False:
        os.mkdir(f"levels-examine/{dname}/{G.name}")
    plt.savefig(f"levels-examine/{dname}/{G.name}/{G.name}.png",bbox_inches='tight')
    plt.close()

    input('Enter any key to continue.')
    print(tabulate(circuitTable,headers=['input','out','aux used','avg cor dist','avg wrong dist','cor < wrong?']))

####################################################################

def dist_between_aux():
    ''' Compare two auxiliary arrays
    '''
    N = 2
    M = 1
    A = 1

    # get spin spaces
    if N not in spin_spaces.keys():
        spin_spaces[N] = ising.get_spin_space(N)

    if M not in spin_spaces.keys():
        spin_spaces[M] = ising.get_spin_space(M)

    if N+M not in spin_spaces.keys():
        spin_spaces[N+M] = ising.get_spin_space(N+M)   

    auxList = [[False, [1,1,1,1]],
               [False, [1,-1,1,-1]],
               [False, [1,-1,-1,1]],
               [False, [1,1,-1,-1]],
               [True, [1,-1,-1,-1]],
               [True, [1,-1,1,1]],
               [True, [1,1,-1,1]],
               [True, [1,1,1,-1]]]
    prefix = 0
    table = []
    # get a distance
    dname,metric = dist_select()

    for taux1 in auxList:
        for taux2 in auxList:
            prefix += 1
    
            # two auxiliary arrays, ought to be one np.array for each input, 
            # so these are length N arrays
            f1, aux1 = taux1
            f2, aux2 = taux2
            
            # reshape aux1 and aux2 so they are array of numpy arrays
            aux1 = np.array(aux1).reshape(2**N,A)
            aux2 = np.array(aux2).reshape(2**N,A)
            
 
            # set x to be decimal representation of spin_space[M] and
            # set y to be array of 0's same length as x
            x = range(0,len(spin_spaces[N+M]))
            y = [0 for a in x]
            D = 0

            for i in range(0,len(spin_spaces[N])):
                # get the auxiliary spin corresponding to input
                a1 = np.array(aux1[i])
                a2 = np.array(aux2[i])
                ispin = spin_spaces[N][i]
                print(ispin)
                # iterate through outputs and populate y with distances
                for ospin in spin_spaces[M]:
                    # set the in/out spin
                    spin = np.concatenate((ispin,ospin))

                    # get the decimal representation of spin
                    num = spinspace.spin2dec(spin)

                    # get the two total spins, i.e. in/out/aux
                    tspin1 = np.concatenate((spin, a1))
                    tspin2 = np.concatenate((spin,a2))
                    y[num] = metric(tspin1,tspin2)
                    D += y[num]

            # readable aux1, aux2
            raux1 = []
            raux2 = []
            for i in range(len(aux1)):
                for j in range(len(aux1[i])):
                    raux1.append(aux1[i][j])
                    raux2.append(aux2[i][j])

            plt.plot(x,y)
            plt.title(f'Distances between spins of length {N+M} for auxiliary arrays\n#1:  {raux1}\n#2: {raux2}')

            # feasibility tag
            ftag = ('F' if f1 else 'I') + 'x' + ('F' if f2 else 'I')

            if os.path.exists("aux-compare") == False:
                os.mkdir("aux-compare")
            if os.path.exists(f"aux-compare/{dname}") == False:
                os.mkdir(f"aux-compare/{dname}")
            if os.path.exists(f"aux-compare/{dname}/A{N},M{M},A{A}") == False:
                os.mkdir(f"aux-compare/{dname}/A{N},M{M},A{A}")
            plt.savefig(f"aux-compare/{dname}/A{N},M{M},A{A}/{prefix}_{ftag}_D{D/N}.png",bbox_inches='tight')
            plt.close()

            table.append([prefix,raux1,raux2,ftag,D/N])

    table.sort(key = lambda x: x[3])
    print(tabulate(table,headers=['#','Aux 1','Aux 2',f'{ftag}','D/N']))

####################################################################

def test_aux_hunt():
    N1 = 1
    A = 1
    G = ising.IXOR(N1=N1,A=A)

    print('===========================================================')
    print(f'Examining minimum ising norms for XOR N = {G.N}, A = {G.A}')
    print('===========================================================\n')
    table = []
    for ispin in spinspace.get_spin_space(G.N):
        a = G.min_norm(ispin)
        table.append(a)

    print(tabulate(table,headers=['spin','min h', 'min J', 'min norm']))

def base_param_hunt():
    N1 = 1
    A = 1
    G = makePICircuit()

    print('===========================================================')
    print(f'Examining minimum ising norms for XOR N = {G.N}, A = {G.A}')
    print('===========================================================\n')
    table = []
    table.append(G.min_separation_base())
    
    print(tabulate(table,headers=['max h','max J','max avg separation']))
####################################################################

def main_program():
    print("EXPLORE ISING DISTANCE")
    print("----------------------\n")
    print("\n\nSelect an option.")
    print("---------------------")
    print("  1. Generate distance histograms")
    print("  2. Tabulate distances for fixed spin")
    print("  3. Examine levels")
    print("  4. Examine auxiliary spins")
    print("  5. Examine norms")
    print("  6. Hunt for good base h and J")

    select = int(input("\nselection: "))
    print()

    if select == 1:
        dist_hist_menu()

    elif select == 2:
        make_dist_data_table()

    elif select == 3:
        dist_between_level()

    elif select == 4:
        dist_between_aux()

    elif select == 5:
        test_aux_hunt()

    elif select == 6:
        base_param_hunt()

    else:
        print("Selection invalid!")

def prompt(message): 
    val = input(message) 
    if val.strip().lower() == 'y': 
       return True 
    elif val.strip().lower() == 'n': 
       return False 
    else: 
       print('Error, please type y or n') 
       prompt(message)

def main():
    while True:
        main_program()
        
        # exit condition 
        if prompt("Continue? (y/n)") == False:
            break

main()
