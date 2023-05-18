import torch
import math
import oneshot
from enum import Enum
from oneshot import MLPoly
from itertools import combinations

from functools import cache


class ReductAlgo(Enum):
    """An easy way to call all reduction methods for any choice of C"""

    Rosenberg = 0
    PositiveFGBZ = 1
    NegativeFGBZ = 2
    FreedmanDrineas = 3

    @staticmethod
    def all_supersets(poly: MLPoly, C: tuple):
        """Returns the maximum set of hyperedges with nonzero weights which have C as a subedge"""
        setC = set(C)
        return tuple(H for H in poly.coeffs if setC.issubset(set(H)))

    def __call__(self, poly: MLPoly, C: tuple):
        """Calls the correct reduction algorithm"""
        if self == ReductAlgo.Rosenberg:
            # do nothing if C is not a choice of quadratic
            if len(C) != 2:
                return poly

            H = ReductAlgo.all_supersets(poly, C)
            M = sum([max(0, poly.get_coeff(key)) for key in H])
            return oneshot.Rosenberg(poly, C, H, M)
        elif self == ReductAlgo.PositiveFGBZ:
            return oneshot.PositiveFGBZ(poly, C, ReductAlgo.all_supersets(poly, C))
        elif self == ReductAlgo.NegativeFGBZ:
            return oneshot.NegativeFGBZ(poly, C, ReductAlgo.all_supersets(poly, C))
        elif self == ReductAlgo.NegativeFGBZ:
            return oneshot.FreedmanDrineas(poly, C, ReductAlgo.all_supersets(poly, C))


class MLPolyEnv:
    """An environment for training an RL.

    Intended Setup
    --------------
        env = MLPolyEnv(1000)
        env.set_params(numvar=100, sparsity=0.1, intcoeff=True, ...)
        env.reset()


    Attributes
    ----------
        maxvar : int
            the maximum number of variables, auxiliary or otherwise
        numaux : int
            the maximum number of auxiliary variables allowed
        numallvar : int
            the maximum number of variables that can be present in a polynomial

    """

    #####################
    ### STATIC PARAMETERS
    #####################

    NOCHANGE_SCORE = 1  # penalty for no-change action
    TERMINAL_DEGREE = 2  # degree at which reduction should stop

    def __init__(self, maxvar, maxaux=-1, maxdeg=5):
        """
        Initializer of the environment. The parameters set here should NEVER CHANGE after initialization, except the flag _isset. Model should be trained on an unchanging set of these parameters.

        Parameters in method `set_params` can be changed during training to train against different type of polynomials

        """
        # set the number of total variables and the number of allowed auxiliaries
        self.maxvar = maxvar
        self.maxaux = math.ceil(0.5 * maxvar) if maxaux == -1 else maxaux
        assert self.maxaux < self.maxvar

        self.maxdeg = maxdeg
        self._isset = False

    ################################
    ### PROPERTIES
    ################################
    @property
    def total_monomials(self):
        """The total number of distinct multilinear monomials possible in this MLPolyEnv"""
        try:
            return self._total_monomials
        except NameError:
            self._total_monomials = sum(
                math.comb(self.maxvar, k) for k in range(self.maxdeg + 1)
            )
            return self._total_monomials

    def allowed_terms(self):
        for k in range(self.maxdeg + 1):
            for term in combinations(list(range(self.maxvar)), r=k):
                yield term

    ################################
    ### ENVIRONMENT SETTINGS
    ################################

    def set_params(
        self,
        numvar=-1,
        mindeg=0,
        mincoeff=-10,
        maxcoeff=10,
        sparsity=0.1,
        intcoeff=False,
    ):
        """Sets the parameters for polynomial generation

        Parameters
        ----------
            numvar : int (default=self.numvar)
                the number of variables in the polynomial. By default it
            mindeg : int
                the minimum possible degree of a term which can appear with nonzero coefficient
            maxdeg : int
                the maximum possible degree of a term which can appear with nonzero coefficient
            mincoeff : float
                the minimum possible coefficient of each monomial
            maxcoeff : float
                the maximum possible coefficient of each monomial
            sparsity : float
                percentage of terms to have nonzero coefficients. Will be sparser once terms outside the degree range are thrown away.
            intcoeff : bool
                restrict to integer coefficients or not
        """
        ########################
        ### perform basic checks
        ########################

        if numvar > self.maxvar:
            raise Exception(
                "Parameter `numvar` cannot be greater than maxvar, maximum number of variables!"
            )

        if numvar == -1:
            numvar = self.maxvar

        if mindeg > self.maxdeg:
            raise Exception("Parameter `mindeg` cannot be greater than `maxdeg`!")

        # ensure there is an actual range from which to select polynomials
        assert mincoeff < maxcoeff

        self.numvar = numvar
        self.mindeg = mindeg
        self.mincoeff = mincoeff
        self.maxcoeff = maxcoeff
        self.sparsity = sparsity
        self.intcoeff = intcoeff
        self.isset = True

    def reset(self):
        """Resets this MLPolyEnv. Does NOT change the parameters controlled by `set_params` method."""
        # run the default parameter setting if user has not set them manually
        if self.isset == False:
            self.set_params()

        self.poly = MLPolyEnv.genpoly(
            numvar=self.numvar,
            mindeg=self.mindeg,
            maxdeg=self.maxdeg,
            mincoeff=self.mincoeff,
            maxcoeff=self.maxcoeff,
            sparsity=self.sparsity,
            intcoeff=self.intcoeff,
        )
        self.score = 0
        self.current_aux = 0
        self._original_numvar = self.poly.num_nonzero_variables()

    ################################
    ### ACTION AND SCORING METHODS
    ################################
    def reduce(self, method: ReductAlgo, C):
        newpoly = method(self.poly, C)
        return self.update_score(newpoly)

    def update_score(self, newpoly: MLPoly):
        """Calculates the current score of the game.

        IN THE FUTURE: Should account for more factors, for instance, the appearance of submodular terms should be penalized, possibly with a weight dependant on their coefficients. This penalty should be calculated relative to the starting submodularity of the polynomial so that bad starting polynomials don't receive artificially bad reduction scores.

        """
        # done if the reduction achieved the desired degree
        done = self.TERMINAL_DEGREE == newpoly.degree()

        # no change
        if self.poly == newpoly:
            self.score += MLPolyEnv.NOCHANGE_SCORE

        else:
            newaux = newpoly.num__nonzero_variables() - self._original_numvar
            self.score += newaux - self.current_aux
            self.current_aux = newaux

        # check if addition of new auxiliaries exceeded the limit
        if self.current_aux > self.maxaux:
            done = True
            self.score += 1000

        return done, self.score

    #############################
    ### STATIC METHODS
    #############################
    @staticmethod
    def genpoly(
        numvar,
        mindeg=0,
        maxdeg=3,
        mincoeff=-10,
        maxcoeff=10,
        sparsity=0.1,
        intcoeff=False,
    ):
        """Generates a random polynomial using the parameters defined in self.set_params

        Parameters
        ----------
            numvar : int (default=self.numvar)
                the number of variables in the polynomial
            mindeg : int
                the minimum possible degree of a term which can appear with nonzero coefficient
            maxdeg : int
                the maximum possible degree of a term which can appear with nonzero coefficient
            mincoeff : float
                the minimum possible coefficient of each monomial
            maxcoeff : float
                the maximum possible coefficient of each monomial
            sparsity : float
                percentage of terms to have nonzero coefficients. Will be sparser once terms outside the degree range are thrown away.
            intcoeff : bool
                restrict to integer coefficients or not
        """

        import random

        diff = maxcoeff - mincoeff

        # generate random coefficients
        gencoeff = (
            lambda: round(diff * random.random() + mincoeff)
            if intcoeff
            else diff * random.random() + mincoeff
        )

        # randomly select the terms which will appear in poly
        num_nonzero = math.floor(sparsity * 2**numvar)
        nonzero = random.sample(list(range(2**numvar)), k=num_nonzero)

        # convert to integers to multi-indices and filter out by degree requirements
        nonzero = [MLPoly.int2term(k) for k in nonzero]
        nonzero = [
            term for term in nonzero if len(term) >= mindeg and len(term) <= maxdeg
        ]

        # convert from integers to terms and assign them coefficients
        coeffs = {MLPoly.bin2term(term): gencoeff() for term in nonzero}

        return MLPoly(coeffs)

    def tensor(self):
        """Returns the polynomial of this MLPolyEnv as a tensor"""
        degtensor = torch.nn.functional.one_hot(
            torch.tensor([self.poly.degree()]), num_classes=self.maxdeg + 1
        )
        polytensor = torch.tensor(
            [self.poly.coeffs[term] for term in self.allowed_terms()]
        )
        return torch.cat((degtensor, polytensor))
