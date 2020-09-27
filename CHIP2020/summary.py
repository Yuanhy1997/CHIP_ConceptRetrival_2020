import os
from load_icd import load_icd
from tqdm import tqdm

with open("train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

x_list = []
y_list = []
for line in lines:
    l = line.strip().split("\t")
    x_list.append(l[0].lower())
    y_list.append(l[1].split("##"))
y_list = [[icd.lower() for icd in y] for y in y_list]

icd2str, match_list = load_icd()

len_x = [len(x) for x in x_list]
len_y = [max([len(one_y) for one_y in y]) for y in y_list]
len_match = [len(match) for match in match_list]
print(max(len_x), max(len_y), max(len_match))

print(max([len(y) for y in y_list]))

"""
103 36 71
16
"""
import numpy as np
print(np.percentile(len_x, 99))
print(np.percentile(len_y, 99))
print(np.percentile(len_match, 99))