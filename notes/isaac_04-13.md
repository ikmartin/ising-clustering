### Two Conjectures: One Soft One Unknown

Let $H_{s,t}$ denote the open halfspace in $\mathbb{R}^{G(G-1)/2}$ defined by the constraint $H(s,f(s)) < H(s,t)$. The solution space of possible $(h,J)$ pairs is then clearly a polyhedral cone with its boundary removed, as it is the intersection of all these halfspaces. The soft conjecture is that the primitive rays of this polyhedral cone are all vectors in $\{-1,0,+1\}^{G(G-1)/2}$. 

I haven't tried to prove this yet, but I'm fairly certain it's true. If I'm correct, then if we weaken the reverse Ising problem by changing our constraints to $H(s,f(s)) \leq H(s,t)$, then all boundary points are suddenly solutions. This means that a solution exists if and only if there is a solution of the form $(h,J) \in \{-1,0,+1\}^{G(G-1)/2}$.

Given such a solution, one can then ask if it can be upgraded to a solution to the original (strong) problem. There could be a scenario in which the answer is no: when the intersection of all the closed half spaces is not empty, but has dimension strictly less than the ambient space. If the solution cone is full dimensional however, then any ball around a weak solution will contain a strong solution.

We must then check to see if the solution cone is ever not full dimensional. My gut says that it will only FAIL to be strong dimensional if there exist two half spaces in the intersection which are reflections of one another, or equivalently whose defining normal vectors are antipodal. This is the second conjecture, and it requires more thought. I think it's also true however: the solution cone will be nonempty but not full dimensional exactly when there is some cone resulting from intersecting some subset of the half spaces which intersects with another half space along their boundary. This "boundary only" intersection can only occur if there were two half spaces which intersected only along their boundary, which can only happen if they are reflections of one another.

This lays out a roadmap to solving the reverse Ising problem with auxiliary spins.

1. Find a weak solution of the form $(h,J) \in \{-1,0,+1\}^{G(G-1)/2}$ using TopDown clustering with a sgn-criterion. Potentially computationally intense. This tells you how to add auxiliary spins to obtain a weak solution. See below for a description of this algorithm, and see the last section for a quick discussion of how clusterings might tell you which auxiliary spins to add.
2. Determine whether the solution cone is full dimensional by checking the normal vectors of all the conditions. This may not be that computationall intense. Comparing two conditions for inputs $s$ and $s'$ looks like this: for all choices of $t$ and $t'$ in output spin space we have $(s,f(s)) - (s,t) = (0,f(s) - t)$ and $(s', f(s')) - (s', t') = (0, f(s') - t')$. We need to ensure that $f(s) - t = f(s') - t'$ if and only if $s = s'$ and $t = t'$, in which case everyhing is zero and hence trivial. If intersections along boundaries do occur, then auxiliary spins can be introduced to easily solve this. You could avoid doing the search by simply adding a single extra auxiliary spin, I think.
3. Once a weak solution has been found and we have ensured we have a full-dimensional solution cone, use gradient descent or something to get a strong solution somewhere in an open ball around the weak solution.

Portion 3 of the above "algorithm" is also unnecessary, for if the two conjectures I mention are true, then 1 and 2 are sufficient since together they would ensure we have a consistent pre-Ising circuit.

### Overview of TopDown Clustering Algorithm

Let $S^N$ and $S^M$ be the input and output spin spaces respectively for a pre Ising circuit $G = N \cup M$ without auxiliary spins whose logic function is $f:S^N \to S^M$. A **top-down clustering algorithm** on $G$ produces a clustering of $S^N$ such that, within each cluster, all input levels can be simultaneously satisfied. Such an algorithm clusters the inputs of $G$ into groups for which the optimization problem can be solved.

It's not hard to produce such algorithms: the most naive choice would be to cluster $S^N$ into $2^N$ clusters, i.e. into singletons. Each input level $L_s$ is satisfied by setting $(h,J) = -\nu(s, f(s))$,  where $\nu: S^G\to S^{G(G-1)/2}$ is the embedding of $S^G$ into virtual spin space. Another naive choice would be to cluster $S^N$ into pairs of spins, as two inputs levels can (almost) always be simultaneously solved.

Thus, the goal of a good top-down clustering algorithm should be to *minimize the number of clusters*. Here's an example algorithm which gets a little closer to that goal than the naive choices above. The basic ideal is to start with $S^N$ and split it up, or *refine* it, into smaller clusters until some criterion is met. That is, start from the top, and work your way down.

```pseudocode
function refine_criterion(cluster) -> bool:
	"""Takes a cluster and decides whether it needs to be refined or not"""

function refine(cluster) -> list[Cluster]:
	"""Takes a cluster and splits that cluster into smaller clusters"""

function topdown(spinspace) -> list[Cluster]:
	# initialize the clustering
	clusters = [spinspace]
	complete = False
	# main loop
	while complete is False:
		complete = True
		for cluster in clusters:
			
			if refine_criterion(cluster) is True:
				complete = False # we need further refinement
				
				# replace the failed cluster with a refinement
				clusters.remove(cluster)
				cluster.append(refine(cluster))

	return clusters
```

There are two additional functions which need to be implemented: `refine` and `refine_criterion`. Let's discuss the latter first. If we have another function `ham_vec` which can take as input a set of spins and outputs a deterministic choice of an hJ vector for the Hamiltonian, then one implementation of refine_criterion might be this:

```pseudocode
function refine_criterion(cluster) -> bool:
	vec = ham_vec(cluster)
	
	if vec satisfies all spin in cluster:
		return False # no further refinement needed
	else: 
		return True # need to refine further
```

I've explored two choices for `ham_vec`, one returns the Q-vector `qvec` of `cluster` and the other returns the "sign" `sgn(qvec)` of the Q-vector. The latter seems to be a better choice, given the discussion above. I'd also like to lossen the refine criterion to "weakly satisfies all spin" given the conjectures from the beginning.

Next is `refine`. There are still a lot of open questions regarding its implementation, and several distinct approaches one might take. Is it advantageous to split a cluster up into a few clusters and hope that they are satisfied, or is it better to split it into many clusters immediately which have a high chance of being satisfied?

Currently, all implementations I've explored for `refine` progress in the same way.

1. Choose two points $c_1$ and $c_2$ from the cluster to be the centers of two new clusters $C_1$ and $C_2$. This can be done randomly, or more advantageously by choosing them to be as far apart as possible. I've been doing the latter by choosing $c_1$ and $c_2$ to be the points which maximize hamming distance (in virtual spin space, so second order Hamming distance) between spins in the cluster.
2. Split the cluster into two new clusters by assigning points to either $C_1$ or $C_2$ using proximity to the cluster centers in virtual spin space. If a spin is equidistant from both centers, save it for later by storing it in a `ties` list.
3. Break ties using some mix of protocols.

### The Bug

There is typically an element of randomness involved in either choosing centers or in breaking ties, and this means that two iterations through the `topdown` algoirithm will produce different clusterings. There exist no 2-clusterings for MUL2x2 using a "sgn" version of `refine_criterion`, I checked them all. There exist 15,851 distinct 3-clusterings using `sgn_refine_criterion`. The only implementation of the algorithm that has been able to find any of these 3-clusterings is not one I intentionally designed, it resulted from a bug in my implementation of `refine`. When choosing the centers $c_1$ and $c_2$, rather than checking distance between input/output pairs, I *only* checked for distance between inputs. This means I was only comparing the hamming distance between the input spins and the pairwise products of the input spins. This produced some of the worst clusterings (largest number of sets in the partition) but also the best I've seen: it recovered a 3-clustering once in every 150 runs or so. I can't yet explain why this is.

Without this bug, the algorithm can frequently find 4-clusterings but has yet to find a 3-clustering using either `sgn` or `qvec` refine criterions.

### Clusterings and Auxiliary Spins

A choice of auxiliary arrays for a pre-Ising graph $G$ is simply a choice of $A$ and a choice of function $g:S^N \to S^A$. We can therefore think of an auxiliary array as a function, $g$.

Thinking this way, it is clear that each auxiliary array gives us a clustering of $S^N$ by taking inverse images of auxiliary spins:
$$
S^N = \bigcup_{\alpha \in S^A} g^{-1}(\alpha)
$$
Such a clustering groups input spins with other input spins sharing the same auxiliary spin state.

This motivate the idea of top-down clustering: cluster input spins and then assign a distinct auxiliary spin state to each cluster. Does this produce a feasible auxiliary array? That's basically the question I'm trying to answer. The difficulty in this approach is wrapped up in deciding what the proper implementations of `refine` and `refine_criterion` ought to be. Using the `refine_criterion` implementation I described above, it's particularly dependent on what choice for `ham_vec` one uses.

My hope and dream is to find a `ham_vec` implementation which associates a vector valued in $\{-1,0,+1\}$ to each `cluster` and then, only requiring that the non-strict conditions are satisfied, obtain a clustering algorithm which achieves the following:

1. produces a (small) clustering which,
2. after assigning unique auxiliary states to each clustering,
3. terminates in zero steps upon rerunning with axuiliary states added.

If such an algorithm exists, and both of my above conjectures are true, then this could solve reverse Ising. I think finding this algorithm is the harder part, both of the conjectures are true -- I'm between 53% and 97% sure.

### Another way to produce a ham_vec valued in $\{-1, 0, +1\}$

Fix a spin $s \in S^N$. The normal vector $v(s,t)$ to the half space $H(s,f(s)) < H(s,t)$ satisfies the inequality, of course. Define $P_a = \operatorname{sgn}\big(\sum_{t \neq f(s)} v(s,t)\big)$. How's that work as a `refine_criterion`?