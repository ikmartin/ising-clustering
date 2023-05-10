### Thoughts following lvec breakthrough and second refine_criterion check

Our current `refine_criterion` returns `False` if there exists an `hvec` which simultaneously solves all inputs in the cluster, and `True` otherwise. There is good empirical evidence to suggest that this is the correct thing to think about, we've tested it on every cluster arising from feasible auxiliary arrays in the 2x2x1 and 2x3x1 cases and every single clustering is terminal, in the sense that an iterative clustering algorithm using this refine criterion would seek no further refinement. We should try this on the available data for the 3x3x3, but we haven't gotten to that yet.

This evidence suggested that this `refine_criterion` is at least necessary, but it does not answer sufficiency. If you believe it is also sufficient, then you should seek ways to refine your clusters in order to maximize the chance that all inputs are simultaneously solvable.

To do this, one might think about the likelihood that a random choice of $h,J$ satis