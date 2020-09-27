import os
from load_icd import load_icd, load_additional_icd, clean
from tqdm import tqdm
import pkuseg


seg = pkuseg.pkuseg(model_name='medicine')

with open("train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

x_list = []
y_list = []
for line in lines:
    l = line.strip().split("\t")
    x_list.append(clean(l[0]))
    y_list.append(l[1].split("##"))
y_list = [[clean(icd) for icd in y] for y in y_list]

icd2str, match_list = load_icd()
str2standard = load_additional_icd()
print(match_list[0:5])

import pickle
with open('co_exist_tfidf_lenth>1.pkl', 'rb') as f:
    co_exist = pickle.load(f)

pairs = {}
for key in tqdm(list(co_exist.keys())):
    pair = key.split('-')
    if co_exist[key] > 500 and pair[0] in pairs.keys():
        pairs[pair[0]].append(pair[1])
    if co_exist[key] > 500 and pair[0] not in pairs.keys():
        pairs[pair[0]] = [pair[1]]

def acc(pred_y, true_y):
    term_count = 0
    predict_count = 0
    correct_count = 0
    exact_count = len(true_y)
    exact_correct = 0
    for i in range(len(true_y)):
        term_count += len(true_y[i])
        predict_count += len(pred_y[i])
        for x in pred_y[i]:
            if x in true_y[i]:
                correct_count += 1
        if set(pred_y[i]) == set(true_y[i]):
            exact_correct += 1
    print(f"Term count: {term_count}")
    print(f"Term predict: {predict_count}")
    print(f"Term correct: {correct_count}")
    print(f"Term recall: {correct_count / term_count}")
    print(f"Term precision: {correct_count / predict_count}")
    print("---")
    print(f"All count: {exact_count}")
    print(f"Acc: {exact_correct / exact_count}")
    return exact_correct / exact_count

punc_set = set("、，。,.；;()（）[]【】\"\'")
# consider split punctuation
def split(x, punc_set):
    opt = []
    now = ""
    for ch in x:
        if ch in punc_set:
            if now:
                opt.append(now)
                now = ""
        else:
            now = now + ch
    if now:
        opt.append(now)
    return opt

def set_find(x, y):
    use_punc_set = set([punc for punc in punc_set if not punc in y])
    split_x = split(x, use_punc_set)
    for xx in split_x:
        if set(y) <= set(xx):
            return True
        if len(set(y) & set(x)) > 3:
            return True
    return False

def set_match(x_list):
    pred_y = []
    for x in tqdm(x_list):
        now_y = []
        for y in match_list:
            if set(x)&set(y):
                if set_find(x, y):
                    now_y.append(y)
        for y, standard_y in str2standard.items():
            if set(x)&set(y):
                if set_find(x, y):
                    now_y.append(standard_y)
        x_l = seg.cut(x)
        for w in x_l:
            if w in pairs.keys():
                now_y = now_y + pairs[w]
        pred_y.append(now_y)
    return pred_y

pred_y = set_match(x_list)
print(acc(pred_y, y_list))

"""
Term count: 14490
Term predict: 13417
Term correct: 5195
Term recall: 0.3585231193926846
Term precision: 0.3871953491838712
---
All count: 8000
Acc: 0.072625
0.072625


交集数量>3
Term count: 14490
Term predict: 1316117
Term correct: 10837
Term recall: 0.7478951000690132
Term precision: 0.008234070375202204
---
All count: 8000
Acc: 0.008625
0.008625

交集数量>4
Term count: 14490
Term predict: 353957
Term correct: 8492
Term recall: 0.5860593512767426
Term precision: 0.023991614800667877
---
All count: 8000
Acc: 0.0195
0.0195


Term count: 14490
Term predict: 353957
Term correct: 8492
Term recall: 0.5860593512767426
Term precision: 0.023991614800667877
---
All count: 8000
Acc: 0.0195
0.0195
"""

