import os
from load_icd import load_icd, load_additional_icd, clean
from tqdm import tqdm
import pkuseg
import pickle
import char_distance
punc_set = set("、，。,.；;()（）[]【】\"\'")

def dice(x, y):
    intersec = len(x.intersection(y))
    return 2. * intersec / (len(x) + len(y))

def jaccard(x, y):
    unions = len(x.union(y))
    intersec = len(x.intersection(y))
    return intersec / unions


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

def set_find(x, y, metric = 'dice', threshold = 0.3, threshold_set_overlap = 4):
    use_punc_set = set([punc for punc in punc_set if not punc in y])
    split_x = split(x, use_punc_set)
    for xx in split_x:
        if set(y) <= set(xx):
            return True
        if len(set(y) & set(x)) > threshold_set_overlap:
            return True
        if metric == 'dice':
            if dice(set(x), set(y)) > threshold:
                return True
        elif metric == 'jaccard':
            if jaccard(set(x), set(y)) > threshold:
                return True
    return False


def acc(pred_y, true_y):
    term_count = 0
    predict_count = 0
    correct_count = 0
    exact_count = len(true_y)
    exact_correct = 0
    for i in range(len(true_y)):
        term_count += len(true_y[i])
        predict_count += len(set(pred_y[i]))
        for x in set(pred_y[i]):
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

def set_match(x_list, 
            y_train_list, 
            icd2str, 
            match_list, 
            str2standard, 
            word_segtool, 
            distance_metrics = 'dice', 
            threshold_metrics = 0.3, 
            threshold_set_overlap = 4, 
            co_exist_pairs = False, 
            wd = False):
    # x_list = x_list[:10]


    pred_y = []
    for x in tqdm(x_list):
        now_y = []
        for idx, y in enumerate(match_list):
            if set(x)&set(y):
                if set_find(x, y, distance_metrics, threshold_metrics, threshold_set_overlap):
                    now_y.append(y)
        for y, standard_y in str2standard.items():
            if set(x)&set(y):
                if set_find(x, y, distance_metrics, threshold_metrics, threshold_set_overlap):
                    now_y.append(standard_y)
        if co_exist_pairs:
            x_l = seg.cut(x)
            for w in x_l:
                if w in co_exist_pairs.keys():
                    now_y = now_y + co_exist_pairs[w]
        if wd:
            now_y = now_y + wd[idx]

        pred_y.append(list(set(now_y)))
    print('Distance metric:\t'+distance_metrics)
    print(acc(pred_y, y_train_list))
    return pred_y


if __name__ == '__main__':
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

    seg = pkuseg.pkuseg(model_name='medicine') 
    with open('co_exist_tfidf_lenth>1.pkl', 'rb') as f:
        co_exist = pickle.load(f)
    pairs = {}
    for key in tqdm(list(co_exist.keys())):
        pair = key.split('-')
        if co_exist[key] > 500 and pair[0] in pairs.keys():
            pairs[pair[0]].append(pair[1])
        if co_exist[key] > 500 and pair[0] not in pairs.keys():
            pairs[pair[0]] = [pair[1]]

    if True:
        wd_out_list = char_distance.find_distance_xy(x_list)
    else:
        wd_out_list = False

    set_match(x_list, y_list, icd2str, match_list, str2standard, word_segtool = seg ,
            distance_metrics = 'dice',
            co_exist_pairs = False, 
            wd = wd_out_list)
'''
Distance metric:        dice
Term count: 14490
Term predict: 5575733
Term correct: 12854
Term recall: 0.8870945479641131
Term precision: 0.002305347117589741
---
All count: 8000
Acc: 0.0
0.0
'''