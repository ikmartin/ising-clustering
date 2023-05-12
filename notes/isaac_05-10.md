### Thoughts following lvec breakthrough and second refine_criterion check

Our current `refine_criterion` returns `False` if there exists an `hvec` which simultaneously solves all inputs in the cluster, and `True` otherwise. There is good empirical evidence to suggest that this is the correct thing to think about, we've tested it on every cluster arising from feasible auxiliary arrays in the 2x2x1 and 2x3x1 cases and every single clustering is terminal, in the sense that an iterative clustering algorithm using this refine criterion would seek no further refinement. We should try this on the available data for the 3x3x3, but we haven't gotten to that yet.

This evidence suggested that this `refine_criterion` is at least necessary, but it does not answer sufficiency. If you believe it is also sufficient, then you should seek ways to refine your clusters in order to maximize the chance that all inputs are simultaneously solvable.

To do this, one might think about the likelihood that a random choice of $h,J$ satisfies two inputs $s$ and $s'$. This likelihood is related to the measure of the intersection $\sigma_s \cap \sigma_{s'}$ in the following way:

1. Denote by $S^d$ the unit sphere in $\mathbb R^d$, where $d = \#(h,J)$.
2. Define the measure $\mu$ on $S^d$ to be normalized Lebesgue measure.
3. Then $\mu(\sigma_s \cap \sigma_{s'}\cap S^d)$ is the probability that an $h,J$ vector uniformly sampled from the surface of $S^d$ satisfies both $s$ and $s'$.

Define $m(s,s') = \mu(\sigma_s \cap \sigma_{s'}\cap S^d)$ for convenience. By sampling Gaussian normal vectors in $\mathbb R^d$ 1,000,000 times and checking for simultaneous satisfiability, I was able to estimate $m(s,s')$ for every $s\in \Sigma^N$. This is in a csv file found in ising-clustering/data. It took approximately 8 hours.

However, there's a faster way to approximate the same information. Let $\ell(s)$ be the L-vector of input level $s$, defined to be the sum of all *normalized* row vectors appearing in the constraint matrix of input level $s$. That is,
$$
\ell(s) = \sum_{t \neq f(s)} \frac{\nu(s,t) - \nu(s,f(s))}{\|\nu(s,t) - \nu(s,f(s))\|},
$$
where $t$ iterates over output spinspace $\Sigma^M$ and $f(s)$ is the correct output. This vector always lives inside $\sigma_s$, the solution cone of input $s$, and is a good proxy for the "average" direction of the cone. Hell, it might even be the average direction; I haven't done that calculation though.

Comparing $\ell(s)$ and $\ell(s')$ is a remarkably good way to estimate $m(s,s')$, as illustrated in the google sheet "input_level_intersection_IMul2x3x0_rep=1000000_deg=2_compared_with_lvec", [link here](https://docs.google.com/spreadsheets/d/15aknAcvKtNEdLQ9b3nNSy0nTWFYL3pa5O9nybfbEE78/edit?usp=sharing). Both Euclidean distance and cosine similarity seem to give good proxies. The idea is "similar lvecs $\iff$ large intersection $m(s,s')$."

This allows us to quickly run clustering algorithms which refine based on "solvability likelihood", which remember, is what we're trying to do since we believe in our refine_criterion. Doing centers-based clustering using popular-centers with Euclidean distance between lvecs as our distance yields a deterministic 2-clustering on MUL2x3. Unfortunately, the cluster is not feasible, *despite* passing refine criterion.

This indicates that there is something wrong with our refine criterion, or rather, that it is not *sufficient*. It begs the question, if I have a clustering such that each cluster is simultaneously solvable, what additional assumption will guarantee that the addition of aux will be able to tape the clusters back together?