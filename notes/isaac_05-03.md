### Thinking about refine method

The current refine method chooses two spins which maximize pairwise virtual distance and makes those the centers of the new clusters. This is bad because

1. The pair which maximizes the distance in virtual spin space is not unique. In the case of 2x2, it is quite common for two pairs of input spins to maximize this distance, it's a distance of 20.
2. It often picks out a pair of input spins $(s_1,s_2)$ such that $s_1$ is *close* to many spins and $s_2$ is *far* from many spins.
3. In larger multiplication circuits this refine method fails to produce clusters which can be easily solved with aux values.

However, I think distance in virtual spin space

 