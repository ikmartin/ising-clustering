## Random Constraint Subset Statistics

In order to scale our current Ising algorithms, we need a way to speed up the LP solver. One promising way of achieving this is by only testing solutions on a subset of the constraint set rather than the entire thing.

#### Observation

We have many, many constraints. Many of these are likely to be redundant. We also have many constraints at each input level, and testing all of them seems rather stupid.

#### Ideas

- **Randomly throw away constraints.** While building the solver, there is some chance of simply skipping each constraint.
- **Check only "important" constraints.** Some input levels are harder to satisfy simultaneously than others. Check that more of the constraints in those input levels are satisfied and check fewer of the "good" input levels.
- **Check more constraints involving outputs nearby correct outputs**. It is likely harder to satisfy a constraint if it comes from comparing two hamming-distance 1 outputs rather than hamming-distance large outputs. 