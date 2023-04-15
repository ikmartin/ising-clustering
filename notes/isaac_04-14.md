### Thoughts on `refine criterion`

Qvec and Sgn are both bad `refine_criterion`. When combined with the auxiliary spins, the clusterings they produce get *bigger*, not smaller.

While they seem to be alright choices to generally minimize the collective energy of all inputs, they don't actually make inputs the minimium of their levels -- there is always some random incorrect output which performs better than the correct one.

What I need is a way to quickly read off the points on the hypercube in virtual space which lie in the intersection of a bunch of level cones. Those are the chocies of ham_vec that will actually perform well.