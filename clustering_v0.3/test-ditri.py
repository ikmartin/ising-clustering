import numpy as np
import random as rand
from mathtools import trinum, ditri_array

N = 10
a = [i + 1 for i in range(N)]
b = [-(i + 1) for i in range(trinum(N - 1))]

# print(diag_and_triu_to_array(a, b))

l = lambda N, i: trinum(N) - trinum(N - i) - i
d = lambda N, i: (N - 1) - i

# the array we're testing
data = ditri_array(a, b)

print("\n------------------------------")
print("      TEST OF ditri_array")
print("------------------------------")

# set flag for visualization
vis = input("Visualize? (y/n): ") in ["Y", "y"]

# perform elementwise comparison of data with a and b
print("\n  Elementwise check...")

# current position
pos = 0
for i in range(N):
    # current row length
    rowlen = N - i
    x = l(N, i)
    y = x + d(N, i)
    row = b[x:y]

    if rowlen - 1 != len(row):
        raise ValueError(
            "EXPECTED len(b[x:y]) = {}, got {}".format(rowlen - 1, len(row))
        )

    if vis:
        print(
            "   {} a[{}] = {}, {} = data[{}]".format(
                "X" if a[i] != data[pos] else "y", i, a[i], data[pos], pos
            )
        )
    # check diagonal element
    if a[i] != data[pos]:
        raise ValueError(
            "  ERROR DIAG: expected {} got {}. Index = {}".format(
                a[i], data[pos], (pos)
            )
        )

    # advance to next position
    pos += 1

    # check triangular elements
    for j in range(0, rowlen - 1):
        if row[j] != data[pos]:
            raise ValueError(
                "  ERROR UTRI: expected {} got {}. Index = {}".format(
                    row[j], data[pos], (pos)
                )
            )

        if vis:
            print(
                "   {} b[{}] = {}, {} = data[{}]".format(
                    "X" if b[x + j] != data[pos] else "y",
                    x + j,
                    b[x + j],
                    data[pos],
                    pos,
                )
            )

        pos += 1

if input(("\n\nNumpy vis?")) not in ["Y", "y"]:
    quit()

upper = np.zeros((N, N))
test = np.zeros((N, N))

ind = lambda N, i, j: trinum(N) - trinum(N - i) + j
for i in range(N):
    x = l(N, i)
    y = x + d(N, i)
    upper[i, i:] = [a[i]] + b[x:y]
    for j in range(i, N):
        test[i, j] = data[ind(N, i, j - i)]


print("EXPECTED:")
print(upper)
print("\nRECEIVED:")
print(test)
print("\n-----------------------\n")

print("THESE ARE {}".format("EQUAL" if np.array_equal(upper, test) else "!NOT! EQUAL"))
