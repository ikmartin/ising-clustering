### Best Clustering thus far found

```
RESULT OF TopDownFarthestPair on Mul2x2
---------------------------------------
    1
       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]   center: -1
    2
       [0, 1, 2, 4, 8, 9, 10, 11, 12, 15]   center: 0
       [7, 3, 5, 6, 13, 14]   center: 7
    3
       [0, 1, 2, 8]   center: 0
       [11, 4, 9, 10, 12, 15]   center: 11
       [7, 3, 5, 6, 13, 14]   center: 7
    Number of generations: 3
    Final number of clusters: 3
    cluster 0: [0, 1, 2, 8]
    cluster 1: [11, 4, 9, 10, 12, 15]
    cluster 2: [7, 3, 5, 6, 13, 14]
```

This was done using `sgn` as the `ham_vec` and `FarthestPair` refinement.



### Comparison of terminal output for break ties experiment:

Spins are frequently equally close to both chosen centers. I thought breaking these ties by minimizing the average pairwise distance within a cluster would be a good idea -- but it turned out to produce consistently bad clusterings. I'm trying to understand why.

Run 1:

```
Generation 2
bin1: [0, 1, 2, 3, 4, 8, 9, 12]
bin2: [7, 5, 6, 13]
ties [10, 11, 14, 15]
...
Number of generations: 5
Final number of clusters: 7
 cluster 0: [0, 1, 2, 3, 4, 8, 12]
 cluster 1: [14, 6]
 cluster 2: [9, 11]
 cluster 3: [10]
 cluster 4: [15]
 cluster 5: [7]
 cluster 6: [5, 13]
```

Run 2:

```
Generation 2
bin1: [0, 1, 2, 3, 4, 8, 9, 12]
bin2: [7, 5, 6, 13]
ties [10, 11, 14, 15]
...
Number of generations: 5
Final number of clusters: 7
 cluster 0: [0, 1, 2, 3, 4, 8, 12]
 cluster 1: [14, 6]
 cluster 2: [9, 11]
 cluster 3: [10]
 cluster 4: [15]
 cluster 5: [7]
 cluster 6: [5, 13]
```

Run 3:

```
Generation 2
bin1: [0, 1, 2, 3, 4, 8, 9, 12]
bin2: [7, 5, 6, 13]
ties [10, 11, 14, 15]
...
Number of generations: 5
Final number of clusters: 7
 cluster 0: [0, 1, 2, 3, 4, 8, 12]
 cluster 1: [14, 6]
 cluster 2: [9]
 cluster 3: [10, 11]
 cluster 4: [15]
 cluster 5: [7]
 cluster 6: [5, 13]
```

Getting same result each time. This is not unexpected, the only chance for randomness is if the average distances within clusters is equal at the "break ties" step.

Here is the result of one run of example 7. The only change is that qvec is used instead of sgn. We get a slightly better result, but it's still bad. All observed clusterings terminated with 6 total clusters.

```
Generation 2
bin1: [0, 1, 2, 3, 4, 8, 9, 12]
bin2: [7, 5, 6, 13]
ties [10, 11, 14, 15]
Generation 3
bin1: [0, 1, 2, 3, 4, 8, 12]
bin2: [11, 9, 10]
ties []
Generation 3
bin1: [7, 5, 13, 15]
bin2: [14]
ties [6]
Generation 4
bin1: [7, 5, 13]
bin2: [15]
ties []
Generation 5
bin1: [7]
bin2: [5]
ties [13]
RESULT OF TopDownBreakTies on Mul2x2 Example 7
--------------------------------------------------
Number of generations: 5
Final number of clusters: 6
 cluster 0: [0, 1, 2, 3, 4, 8, 12]
 cluster 1: [11, 9, 10]
 cluster 2: [14, 6]
 cluster 3: [15]
 cluster 4: [7]
 cluster 5: [5, 13]
Generation progression:
----------------------
  1
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]   center: None    x
  2
    [0, 1, 2, 3, 4, 8, 9, 12, 10, 11]   center: 0    x
    [7, 5, 6, 13, 14, 15]   center: 7    x
  4
    [0, 1, 2, 3, 4, 8, 12]   center: 0    ✓
    [11, 9, 10]   center: 11    ✓
    [7, 5, 13, 15]   center: 7    x
    [14, 6]   center: 14    ✓
  2
    [7, 5, 13]   center: 7    x
    [15]   center: 15    ✓
  2
    [7]   center: 7    ✓
    [5, 13]   center: 5    ✓
```



### Bizarre Oddity That is F!@#ing Stupid

I included a bug in my code at some point. Originally, the code for cluster refinement with random tie breaking computed vdistances between *only* the input portion of spins, ignoring the output. Infuriatingly, this achieved the best clustering score of 3 total clusters (lower is better). That's how the output at the top of this document was achieved. I reintroduced the bug and got the following output:

```
RESULT OF TopDownFarthestPair on Mul2x2
---------------------------------------
Number of generations: 3
Final number of clusters: 3
 cluster 0: [7, 3, 5, 6, 10, 12, 14]
 cluster 1: [0, 1, 2, 4, 8]
 cluster 2: [11, 9, 13, 15]
Generation progression:
----------------------
  1
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]   center: None    x
  2
    [0, 1, 2, 4, 8, 9, 11, 13, 15]   center: 0    x
    [7, 3, 5, 6, 10, 12, 14]   center: 7    ✓
  2
    [0, 1, 2, 4, 8]   center: 0    ✓
    [11, 9, 13, 15]   center: 11    ✓
```

Notice that we have the same centers as before, but the clusters are different. One particular oddity here is that 3 is not clustered with 0, 1, 2, 4 and 8. When the bug is fixed, it is clustered with these spins, which is to be expected since the output of these all is 0. The spins 1, 2, 4 and 8 all have exactly one occurance of 1 in their input, whereas 3 has two ones. Hence their input is not similar at all, even though their output is identical.

For some reason, sticking 3 and 7 together produces a fantastic cluster. This choice is not picked out by the un-bugged clustering method.

### Other Analysis

One might wonder: if a list of spins is satisfied by sgn or qvec, is it still satisfied when I remove some of the spins? The answer is no -- here's a counterexample. This terminal output was obtained from the spin_util.py script.

```
spins: [11 9 13 15]
  all levels satisfied for sgn.
  all levels satisfied for qvec.
  max distance is 20 achieved by
    [(11, 13), (11, 15), (9, 15)]
  diameter is 20
------------------------------------------------------------
SUBSETS LENGTH 3
spins: [11, 9, 13]
  levels NOT satisfied for sgn. Here are the spins that fail:
    13 : [ 1  1 -1  1]
  all levels satisfied for qvec.
spins: [9, 13, 15]
  levels NOT satisfied for sgn. Here are the spins that fail:
    15 : [1 1 1 1]
  all levels satisfied for qvec.
```

Funnily enough, qvec does actually satisfy all of these subsets. I'll start looking for a qvec counterexample too. However, I wonder if one can prove that qvec is actually demonstrably better than sgn, meaning that if qvec fails for a subset so does sgn. If sgn satisfies a level does that imply qvec also satisfies it? I wonder.
