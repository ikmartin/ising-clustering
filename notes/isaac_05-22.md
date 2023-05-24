## Summary of Approaches

Andrew and I have worked on a few different approaches thus far. We've reached a moment of impasse, so it seems worthwhile to summarize.

An important thing to note is that we now play with two different constraint sets:

- **Weak constraints:** the original constraint set, demanding that $H(s_{\text{correct}}) < H(s_{\text{wrong}})$. This is exactly the constraints needed to guarantee that correct answers are the global minima of input levels.
- **Hypercube constraints:** fix an input $s \in \Sigma^N$ and let $f(s) = t_0 \in \Sigma^M$ be the correct output on input level $s$. Let $\Sigma^M(s)(k)$ be the set of incorrect outputs which are exactly hamming distance $k$ from $t_0$. We then demand that $H(s,t_i) < H(s,t_{i+1})$ for all $t_i \in \Sigma^M(s)(i)$ and $t_{i+1} \in \Sigma^M(s)(i+1)$.

### Clustering

*Goal*: Group input spins together into clusters $C_1,...,C_k$ such that

- **Satisfiability:** all inputs in $C_i$ are simultaneously satisfiable
- **Recombination:** by assigning an auxiliary value $\alpha_i$ to all inputs in $C_i$, the circuit becomes solvable.

Our current methods are quite good at achieving satisfiability, they are bad at recombination. All initial effort was put into finding minimal satisfiable clusters, but only recent efforts were spent trying to design for recombination.

The current best method is the *lvec* method. The idea is this: cluster inputs together if they are more likely to be satisfied with the same auxiliary spin and separate them if they are more likely to be satisfied apart. I'm calling this idea "likelihood". To be more concrete: consider a function $L_+:\mathcal{P}(\Sigma^N) \to [0,1]$ such that $L(C)$ is the probability that the inputs in $C$ are all satisfied by assigning them all the same auxiliary. This should be the area of the spherical polyhedra on the surface of a certain hypersphere enclosed by the cone defined by the constraints of the spins in $C$.

More generally, we'd like to somehow capture probability of satisfiability of assigning the same aux to inputs and opposite aux to inputs. To do this, define
$$
\mathcal{C} = \{ (C_1,C_2) \in \mathcal{P}(\Sigma^N) \times \mathcal{P}(\Sigma^N) ~\mid~ C_1 \cap C_2 = \emptyset \}
$$
to be the set of all pairs of subsets of input spins which don't intersect. Now define $L_+:\mathcal{C} \to [0,1]$ to be the likelihood that $C_1, C_2$ are satisfied by the same choice of auxiliary spin and $L_-:\mathcal{C} \to [0,1]$ to be the likelihood that $C_1, C_2$ are satisfied by opposite choices of auxiliary spins. Each time we refine a cluster $C$ into $C_1$ and $C_2$, we're making a guess that the spins in $C_1$ deserve one auxiliary value and the spins in $C_2$ deserve another, so we *ought* to try and maximize $L_-$. We can also combine these two likelihood functions: $L(C_1, C_2) := L_+(C_1,C_2) - L_-(C_1,C_2) \in [-1,1]$. Now $L(C_1,C_2) < 0$ means that $C_1,C_2$ are more likely to be satisfied with opposite auxiliary spins and $L(C_1,C_2) > 0$ means that they are more likely to be satisfied with matching auxiliary spins.

The trouble is that $L$ is easy to calculate (or at least approximate) when $C_1$ and $C_2$ are both singletons. In this case you are only asking for the probability of simultaneous satisfaction for two individual inputs, "pairwise likelihood". You can estimate it by lvec difference: calculate the lvec of both spins with matching temporary aux values and with opposing temporary aux values and then look at the difference. Non pairwise likelihood is much harder to approximate, and is the information we actually need.

Two possible routes forward:

1. Try to get higher-order likelihood info from pairwise info. Stick a graph structure on the inputs and start removing edges according to pairwise likelihood. *Graph refine method*.
2. Find ways to calculate higher order likelihood directly or through more efficient proxy methods. Can try and do a area calculation for spherical polyhedra, would involve calculating triangulations of these hyperspherical polyhedra into spherical simplicies and then adding up the area of all the parts. Would be quite expensive, so would be better to have some proxy like unto lvec's proxy for the pairwise case.

### Polynomial Fitting

The goal of this approach is to find a polynomial, evaluated over $\{-1,+1\}^G$ which satisfies 

### Polynomial Reduction

Any pseudo boolean polynomial $f$ can be "reduced to a quadratic $f'$" by adding auxiliary variables. By "reduced to a quadratic" we mean that we the minima of the $f'$ match the minima of $f$ in the original variables. This means that 

### Boolean Function Combination

