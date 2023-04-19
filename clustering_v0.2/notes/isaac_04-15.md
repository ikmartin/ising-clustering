### Improving `refine_criterion`

Let $G$ be an Ising graph with input spinspace $S^N$. Define
$$
\sigma_s = \{P \in \mathbb R^{G(G-1)/2} \mid P \text{ satisfies } L_s\}
$$
to be the cone of all $h,J$ solutions which satisfy $L_s$. What I want: a quick way to determine whether
$$
\sigma_{s_1} \cap ... \cap \sigma_{s_\ell}
$$
is nonempty for an arbitrary collection of input spins.

**Thought 1:** It will be easier to determine whether
$$
\sigma_{s_1}\cap ...\cap \sigma_{s_\ell} \cap C
$$
 is nonempty, where $C = \{-1, 0, +1\}^{G(G-1)/2}$ is the set of all corners and mid-points of the hypercube in $(h,J)$-space.  

**Thought 2:** Each 

