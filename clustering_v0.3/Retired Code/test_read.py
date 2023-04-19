import numpy as np


def is_float(string):
    """True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False


data = []
with open("data/all_feasible_MUL_2x2x1.dat", "r") as f:
    d = f.readlines()[0:10]
    for i in d:
        print(i)
        k = i.rstrip().split(",")
        print(k)
        data.append([float(i) if is_float(i) else i for i in k])

data = np.array(data, dtype="O")

print(data[0])
print(len(data))
