import json
import numpy as np
import glob
from os.path import join as j_


exp_name = "exp_otsurv_test"
dir_name = "../result"
c_index_lsit = []
for name in ["BRCA", "BLCA", "LUAD", "STAD", "COADREAD", "KIRC"]:
    print("="*10 + name + "="*10)
    json_path_list = []
    for fold in range(5):
        json_path = glob.glob(j_(dir_name, exp_name, name + "_survival", "k=" + str(fold), "*", "*", "summary.csv.json"))
        if len(json_path) != 1:
            print("Error Fold:" + str(fold))
        json_path_list.append(json_path[0])

    # store all c_index_test values
    c_index_values = []

    # read each JSON file and extract the first value of c_index_test
    for json_path in json_path_list:
        with open(json_path, "r") as file:
            data = json.load(file)
        
        # extract and keep three decimal places
        c_index = round(data["c_index_test"][0], 3)
        c_index_values.append(c_index)

    # calculate mean and standard deviation
    mean_c_index = round(np.mean(c_index_values), 3)
    std_c_index = round(np.std(c_index_values), 3)

    c_index_lsit.append(mean_c_index)

    # output results
    # print("Summary:")
    # print(f"c_index_test values: {c_index_values}")
    print(f"Mean: {mean_c_index}")
    print(f"Standard Deviation: {std_c_index}")
    print()

print("="*10 + "All" + "="*10)
print("Exp: " + exp_name)
print("Mean For All: " + str(round(np.mean(c_index_lsit), 3)))

