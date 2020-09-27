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


def jaccard(x, y):
    unions = len(x.union(y))
    intersec = len(x.intersection(y))
    return intersec / unions

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

def set_find(x, y, jaccard_threshold):
    use_punc_set = set([punc for punc in punc_set if not punc in y])
    split_x = split(x, use_punc_set)
    for xx in split_x:
        if set(y) <= set(xx):
            return True
        if len(set(y) & set(x)) > 4:
            return True
        if jaccard(set(x), set(y)) > jaccard_threshold:
            return True
    return False

def set_match(x_list, jaccard_threshold):
    pred_y = []
    for x in tqdm(x_list):
        now_y = []
        for y in match_list:
            if set(x)&set(y):
                if set_find(x, y, jaccard_threshold):
                    now_y.append(y)
        for y, standard_y in str2standard.items():
            if set(x)&set(y):
                if set_find(x, y, jaccard_threshold):
                    now_y.append(standard_y)
        pred_y.append(now_y)
    return pred_y

pred_y = set_match(x_list, jaccard_threshold = 0.5)
print(acc(pred_y, y_list))



'''
Term count: 14490
Term predict: 356074
Term correct: 8607
Term recall: 0.5939958592132505
Term precision: 0.024171941787381274
---
All count: 8000
Acc: 0.01775
0.01775
'''
