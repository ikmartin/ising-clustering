import numpy as np
import ising

#########################################
### Module Flags and Settings
#########################################

''' Default (pseudo) metric to use on spin space with properties
'''

_dist_name = 'ising'

def get_dist(): return _dist_name
def set_dist(name): 
    global _dist_name
    # ensure name is valid
    if name not in dist_dict:
        print("ERROR: Provided distance not valid!")
        return None
    
    _dist_name = name
    print("Set _dist_name to ", _dist_name)

default_dist = property(get_dist,set_dist)

#########################################
# Function Definitions
#########################################

### Distance Methods ###

def ham_dist(a,b,h=None,J=None):
    '''Returns the hamming distance between two arrays.     
    
    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
    # ensure a and b are same length
    if len(a) != len(b):
        print("ERROR: Lengths of provided arrays to not match!")
        return None
    
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1

    return dist

def ham_dist2(a,b,h=None,J=None):
    '''Returns the 'second-order hamming distance' between two arrays. Calculates hamming distance and then adds the hamming distance between (a^t)a and (b^t)b. Can also think of this as the difference between two Ising Hamiltonian functions where all h and J are set to 1.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
 
    # the distance between a and b
    dist = 0

    # iterate through arrays and calculate distance
    for i in range(len(a)):
        # add 1 to distance whenever positions don't match
        if a[i] != b[i]:
            dist += 1 

        # add 1 to distance whenever product doesn't match
        for j in range(i+1,len(a)):
            if a[i]*a[j] != b[i]*b[j]:
                dist += 1

    return dist

def signed_unit_ising_dist(a,b,h=None,J=None):
    '''Returns the signed ising-distance between two arrays. Equivalent to Hamiltonian distance with h = J = 1.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
 
    # the distance between a and b
    dist = 0

    # iterate through arrays and calculate distance
    for i in range(len(a)):
        # add 
        dist += (a[i]-b[i])/2

        # get second order contribution
        for j in range(i+1,len(a)):
            dist += (a[i]*a[j]-b[i]*b[j])/2

    return dist

def unit_ising_dist(a,b,h=None,J=None):
    '''Returns the ising-distance between two arrays. Equivalent to Hamiltonian distance with h = J = 1.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
 
    return abs(signed_unit_ising_dist(a,b))

####################################################################

def signed_unit_ising_dist(a,b,h=None,J=None):
    '''Returns the signed ising-distance between two arrays. Equivalent to Hamiltonian distance with h = J = 1.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
 
    # the distance between a and b
    dist = 0

    # iterate through arrays and calculate distance
    for i in range(len(a)):
        # add 
        dist += (a[i]-b[i])/2

        # get second order contribution
        for j in range(i+1,len(a)):
            dist += (a[i]*a[j]-b[i]*b[j])/2

    return dist

####################################################################

def signed_ising_dist(a,b,h,J):
    '''Returns the signed ising-distance between two arrays. Equivalent to Hamiltonian distance with h = J = 1.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
 
    # the distance between a and b
    dist = 0

    # iterate through arrays and calculate distance
    for i in range(len(a)):
        # add 
        dist += h[i]*(a[i]-b[i])/2

        # get second order contribution
        for j in range(i+1,len(a)):
            dist += J[i,j]*(a[i]*a[j]-b[i]*b[j])/2

    return dist

####################################################################

def ising_dist(a,b,h,J):
    '''Returns the ising-distance between two arrays. Equivalent to Hamiltonian distance with h = J = 1.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    *** h: (array) h values
    *** J: (array) J values
    '''
 
    return abs(signed_ising_dist(a,b,h,J))

####################################################################

def ham_ising_dist_short(a,b,h=None,J=None):
    '''Returns mixed hamming ising distance. Calculates hamming distance and then adds the signed 2nd order term from ising_dist. 

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    '''
 
    # the distance between a and b
    dist = 0

    # iterate through arrays and calculate distance
    for i in range(len(a)):
        # add 
        dist += abs((a[i]-b[i])/2)

        # get second order contribution
        for j in range(i+1,len(a)):
            dist += (a[i]*a[j]-b[i]*b[j])/2

    return abs(dist)

def ham_ising_dist(a,b,h=None,J=None):
    ''' Returns hamming distance plus ising_distance. Ensures d(a,b) = 0 iff a == b
    '''

    return ham_dist(a,b) + ising_dist(a,b)

def dist(a,b,h=None,J=None,dist=None,*,debug=False):
    '''Wrapper for metrics on spin space. Implemented to make changing metrics in the future easier.

    Params:
    *** a: (array) the first array
    *** b: (array) the second array
    *** dist (default = default_dist): (string) the distance method to use

    Optional Params:
    *** debug (default = False): (bool) print debugging messages
    '''
    
    # adding this check allows us to change default_dist later
    if dist is None:
        dist = default_dist

    # print debug statements
    if debug:
        print("Distance method is ",dist)
    # ensure a and b are same length
    if len(a) != len(b):
        print("ERROR: Lengths of provided arrays to not match!")
        return None
    
    return dist_dict[dist](a,b,h=h,J=J)

# for use with experimental distance below
p = 25
h = np.ones(p)
J = np.ones(p**2).reshape(p,p)

for i in range(p):
    if i % 5 == 0:
        h[i] *= -1
    for j in range(i+1,p):
        if i % 3 == 0:
            J[i,j] *= -1

def weird_distance(a,b,h=None,J=None):
    ''' Whatever experimental distance I'm currently using
    '''

    # the distance between a and b
    dist1 = 0
    dist2 = 0

    # iterate through arrays and calculate distance
    for i in range(len(a)):
        # add 
        dist1 += (a[i]-b[i])/2
        dist2 += -(a[i]-b[i])/2

        # get second order contribution
        for j in range(i+1,len(a)):
            dist1 += J[i,j]*(a[i]*a[j]-b[i]*b[j])/2
            dist2 += -(a[i]*a[j]-b[i]*b[j])/2

    return (abs(dist1) + abs(dist2))/2

''' Dictionary containing all available metrics
'''
dist_dict = {
        'hamming' : ham_dist,
        'hamming2' : ham_dist2,
        'unit-ising' : unit_ising_dist,
        'signed-unit-ising' : signed_unit_ising_dist,
        'ising' : ising_dist,
        'signed-ising' : signed_ising_dist,
        'ham-ising-short' : ham_ising_dist_short,
        'ham-ising' : ham_ising_dist,
        'weird-distance' : weird_distance
    }

def get_dist_func(key):

    ''' Helper function for dist_dict. Returns the distance metric corresponding to the provided key.
    
    Returns: key (str), func (method): the name of the distance function and the distance function itself

    Params
    *** key: (int) or (str) the index or key used to retrieve the distance name
    '''

    if type(key) is int:
        name = list(dist_dict.keys())[key]
        return name, dist_dict[name]

    elif type(key) is str:
        return key, dist_dict[key]

    else:
        print("ERROR: need either int or str. Received")
        print(key)
        print('instead')
        return None

#########################################
### Norm Methods
#########################################

def unit_ising_norm(s):
    ''' Returns the ising norm of the provided spin. Simulates Hamiltonian in the case that all h and J parameters are equal to 1.

    Returns: (int) the norm of the provided spin

    Params:
    *** s: (numpy.ndarray) the spin whose norm is to be evaluated.
    '''

    norm = 0
    for i in range(len(s)):
        norm += s[i]
        for j in range(i+1,len(s)):
            norm += s[i]*s[j]

    return norm

####################################################################

def ising_norm(s,h,J):
    ''' Returns the ising norm of the provided spin. Is genuinely just the Hamiltonian.

    Returns: (int) the norm of the provided spin

    Params:
    *** s: (numpy.ndarray) the spin whose norm is to be evaluated.
    *** h: 
    '''

    norm = 0
    for i in range(len(s)):
        norm += h[i]*s[i]
        for j in range(i+1,len(s)):
            norm += J[i,j]*s[i]*s[j]

    return norm

####################################################################
### SUMMED DISTANCE METHODS
####################################################################

def avg_dist_from_center(s,spins,h=None,J=None,*,distkey=None):
    ''' Calculates the average distance from 'center' spin s and list of other spins 'spins'.

    Returns: (float) average distance from spin in spins to 'center' spin s.

    Params:
    *** s: (numpy.ndarray) the center spin
    *** spins: (array(numpy.ndarray)) list of spins whose average distance is calculated
    *** h: (numpy.ndarray) an h array, required if selected distance requires it
    *** J: (numpy.ndarray) a J array, required if selected distance requires it
    *** distkey (default=default_dist): (str or int) key used to indicate descired metric
    '''
    
    # set the metric to use
    if distkey == None:
        distkey = default_dist
    
    dname, metric = get_dist_func(distkey)

    D = 0
    for spin in spins:
        D += metric(spin,s,h=h,J=J)

    return D/len(spins)
