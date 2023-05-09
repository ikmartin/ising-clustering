filename = "prob_correct_MUL_2x3x1_0.999.dat"
remove = [
    "half_feasible_MUL_2x3x1_64_001.out:",
    "half_feasible_MUL_2x3x1_64_002.out:",
    "half_feasible_MUL_2x3x1_128_001.out:",
    "half_feasible_MUL_2x3x1_128_002.out:",
    "half_feasible_MUL_2x3x1_128_003.out:",
    "half_feasible_MUL_2x3x1_128_004.out:",
    "half_feasible_MUL_2x3x1_128_005.out:",
    "half_feasible_MUL_2x3x1_128_006.out:",
    "half_feasible_MUL_2x3x1_128_007.out:",
    "half_feasible_MUL_2x3x1_256_001.out:",
    "half_feasible_MUL_2x3x1_256_002.out:",
    "half_feasible_MUL_2x3x1_256_003.out:",
    "half_feasible_MUL_2x3x1_256_004.out:",
    "half_feasible_MUL_2x3x1_256_005.out:",
    "half_feasible_MUL_2x3x1_256_006.out:",
    "half_feasible_MUL_2x3x1_256_007.out:",
    "half_feasible_MUL_2x3x1_384_002.out:",
    "half_feasible_MUL_2x3x1_1024_01..out:",
    "half_feasible_MUL_2x3x1_1024_001.out:",
    "half_feasible_MUL_2x3x1_1024_002.out:",
    "half_feasible_MUL_2x3x1_1024_003.out:",
    "half_feasible_MUL_2x3x1_1024_004.out:",
    "half_feasible_MUL_2x3x1_1024_005.out:",
    "half_feasible_MUL_2x3x1_1024_006.out:",
    "half_feasible_MUL_2x3x1_1024_007.out:",
    "half_feasible_MUL_2x3x1_1024_008.out:",
    "half_feasible_MUL_2x3x1_1024_009.out:",
    "half_feasible_MUL_2x3x1_1024_010.out:",
    "half_feasible_MUL_2x3x1_1024_011.out:",
    "half_feasible_MUL_2x3x1_1024_012.out:",
    "half_feasible_MUL_2x3x1_1024_013.out:",
    "half_feasible_MUL_2x3x1_2048_001.out:",
    "half_feasible_MUL_2x3x1_2048_002.out:",
    "half_feasible_MUL_2x3x1_2048_003.out:",
    "half_feasible_MUL_2x3x1_2048_004.out:",
    "half_feasible_MUL_2x3x1_2048_005.out:",
    "half_feasible_MUL_2x3x1_2048_006.out:",
    "half_feasible_MUL_2x3x1_2048_007.out:",
    "half_feasible_MUL_2x3x1_2048_008.out:",
    "half_feasible_MUL_2x3x1_2048_009.out:",
    "half_feasible_MUL_2x3x1_2048_010.out:",
    "half_feasible_MUL_2x3x1_2048_011.out:",
    "half_feasible_MUL_2x3x1_2048_012.out:",
    "half_feasible_MUL_2x3x1_2560_001.out:",
    "half_feasible_MUL_2x3x1_2560_002.out:",
    "half_feasible_MUL_2x3x1_3072_001.out:",
    "half_feasible_MUL_2x3x1_3072_002.out:",
    "half_feasible_MUL_2x3x1_4096_001.out:",
    "half_feasible_MUL_2x3x1_8192_001.out:",
]

with open(filename, "r") as file:
    filedata = file.read()

# replace
for thing in remove:
    filedata = filedata.replace(thing, "")

filedata = filedata.replace("x", "1,")
filedata = filedata.replace(".", "-1,")
filedata = filedata.replace(",]", "]")

with open(filename, "w") as file:
    file.write(filedata)
