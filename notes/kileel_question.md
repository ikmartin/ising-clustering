### Hyperplanes on a hypercube question

Let $C = \{-1,0,1\}$ be the subset of points on the $d$-dimensional hypercube in $\mathbb R^d$ whose points are valued in $\{-1, 0, +1\}$. Consider a collection of $d$-dimensional real vectors  $v_i \in C$, together with the associated open halfspaces $H_i = \{x\in \mathbb R^d ~\mid~ x\cdot v_i > 0 \}$ for $1\leq i\leq k$. I'm interested in answering the following question when $0 < d \ll k$, say when $d \approx 258$ and $k\approx 100,000$:

> Is $\bigcap H_i$ nonempty?

I think this is quite computationally expensive to answer, even though it's a linear programming problem. I'm hoping the following two observations could speed up the process:

1. Let $\sigma = \bigcap \overline{H_i}$ denote the closure of $\sigma^\degree$, or the solutions to the loosened constraints $x \cdot v_i \geq 0$. Then $\sigma$ is a polyhedral cone, and   $\bigcap H_i$ is nonempty precisely when $\sigma$ is full dimensional.

This means we can show $\bigcap H_i$ is nonempty by finding a point on the boundary of $\sigma$ and confirming that $\sigma$ is full dimensional. Furthermore, 

2. The primitive generators of $\sigma$ are all points in $C$.

That is, if $\sigma$ is nonempty then it contains a point on it's boundary whose coordinates are all valued in $\{-1, 0, +1\}$. To answer the original question, one could first **try to find a point of $C$ on the boundary of $\sigma$** and then **check whether $\sigma$ is full dimensional**. I'm trying to find an efficient way to check this first condition.

Since determining whether a point $x \in C$ is in $\sigma$ boils down to comparing coordinates between $x$ and all the normal vectors $v_i$, I thought you might be able to sample points from $C \cap \sigma$ using some sort of Metropolis-Hastings algorithm. Perhaps you could start with a graph whose nodes are the points in $C$ and then try to design a stationary probability density such that a randomly initialized point in $C$ steps toward the set $\sigma$. I have no idea how to design such a stationary probability density, however. Have you ever heard of something like this?

