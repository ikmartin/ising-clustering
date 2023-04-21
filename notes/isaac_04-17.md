### Assumption Underlying `TopDown`

The following assumption is made in all current implementations of TopDown:

> Suppose $G$ is an Ising circuit with a known feasible auxiliary array $g:S^N \to S^A$. If $C = \{C_1,...,C_k\}$ is the clustering of $S^N$ resulting from the level sets of $g$, then all inputs in $C_i$ are simultaneously solvable even with auxiliary spins removed.

Not sure how reasonable of an assumption this is, but might be worth trying to prove it. It does give us a way to estimate a value for $A$ though: if $C$ is a k-clustering of $S^N$ then we'll need $|A| = \log_2(k)$.

