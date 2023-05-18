file = open("all_viable_sgn_3-clusters_IMul2x2.dat", "r")
newfile = open("all_viable_sgn_3-clusters_IMul2x2_COPY.dat", "w")

entire_str = file.read()
for line in entire_str.split("]]"):
    newfile.write(line + "]]\n")

file.close()
newfile.close()
