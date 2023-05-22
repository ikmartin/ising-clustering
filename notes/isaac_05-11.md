### Implemented Likelihood Clustering

Today I made a lot of changes to ising.py, fixed some things in spinspace.py and created the first draft of likelihood clustering with aux. Here's the elementary idea behind my new refine_method philosophy:

*Inputs should be clustered together if matching auxiliary spins increase the likelihood that they will be simultaneously satisfied.*

Seems reasonable. Here's the algorithm stated as an optimization problem:

Choose $C_+$ and $C_-$ to be a refinement of a cluster $C$ such that when all inputs in $C_+$ are assigned an auxiliary value of $+1$ and all inputs in $C_-$ are assigned an auxiliary value of $-1$, the probability that a random $h,J$ vector satisfied all inputs in $C$ is maximized.

The problem is then to come up with some efficient proxy for likelihood of simultaneous satisfaction. I can do pairwise satisfiability just fine -- distance between $\ell$-vectors is a good proxy. You can score a choice of auxiliary states by summing up all $\ell$-vectors of elements in the cluster $C$, and you choose between two such choices by selecting the one whose sum is smaller. However -- in all good clusterings of `MUL_2x3x1`, i.e. those coming from feasible auxiliary arrays with good dynamics, 30 and 27 are in the same cluster. This is problematic because 30 and 27 have the largest pairwise distance of any two input spins.