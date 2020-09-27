import os
from load_icd import load_icd, load_additional_icd, clean
from tqdm import tqdm
import pkuseg
import pickle
import numpy as np

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
punc_set = set("`~*&^%$#@!<>?{}、，。,.；;()（）[]【】\"\'")

# consider split punctuation


word_count = {}
word_count_doc = {}
word_tfidf = {}
def count_words(x):
    global word_count
    for word in x:
        if word in word_count.keys():
            word_count[word] += 1
        else:
            word_count[word] = 1

def count_docs(x):
    global word_count_doc
    for word in set(x):
        if word in word_count_doc.keys():
            word_count_doc[word] += 1
        else:
            word_count_doc[word] = 1

def get_word_tfidf(x_list):
    for x in tqdm(x_list):
        x_l = seg.cut(x)
        count_words(x_l)
        count_docs(x_l)
    all_word_num = 0
    for word in list(word_count.keys()):
        all_word_num += word_count[word]
    for word in list(word_count_doc.keys()):
        if word_count_doc[word] > 2:
            word_tfidf[word] = (word_count[word] / all_word_num) * np.log(8000 / (1 + word_count_doc[word]) )

get_word_tfidf(x_list)
sort_word_freq = sorted(word_tfidf.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

#  sort_word_freq[:100]
# [('后', 0.05058048524101326), (',', 0.04690686633648021), ('伴', 0.03339076879374169), 
# ('，', 0.031110821570298482), ('病变', 0.02564231524101387), ('(', 0.025457378591994675), 
# (')', 0.0246637788588689), (';', 0.02370467611069745), ('化疗', 0.022260972984218183),
#  ('不', 0.022260972984218183), ('右侧', 0.021464929325391034), ('、', 0.019327683340902065),
#   ('转移', 0.019079141959070536), ('待', 0.018896230946962952), ('双', 0.01858140220773122), 
#   ('查', 0.01831528760706067), ('中', 0.017374324455692924), ('骨折', 0.017320828278753947), 
#   ('皮肤', 0.016965185163986977), ('分化', 0.016965185163986977), ('及', 0.01682824980603356), 
#   ('慢性', 0.016551517839610702), ('急性', 0.01596441469424319), ('并', 0.015567300046986757), 
#   ('异常', 0.015195514058077671), ('治疗', 0.015022364163293203), ('内', 0.014935470693480696),
#    ('胎儿', 0.014846139874474073), ('感染', 0.014171574388517799), ('周', 0.014115797559467196), 
#    ('功能', 0.014115797559467196), ('（', 0.014054453939025914), ('？', 0.013960467976083385), 
#    ('）', 0.01390352781895191), ('低', 0.013870455952794212), ('多发', 0.0135148032360559), 
#    ('改变', 0.013452029345709668), ('左', 0.013326517956458547), ('1', 0.012623589005883291), 
#    ('肺', 0.012334342766451381), ('左侧', 0.01193627995675674), ('糖尿病', 0.011822961769985292), 
#    ('下肢', 0.011813225524212859), ('全', 0.011745367651872276), ('术后', 0.01142905220957114), 
#    ('双侧', 0.01142905220957114), ('下', 0.011083914064345128), ('软组织', 0.011067758395453495),
#     ('宫颈', 0.01084362843361692), ('2', 0.010772669221780098), ('轻度', 0.01058899033704432), 
#     ('障碍', 0.010559858759328073), ('-', 0.010474626611038346), ('重度', 0.010459856308543779), 
#     ('高血压', 0.010446907508922994), ('缺损', 0.010157768954091184), ('疾病', 0.01005636519087096), 
#     ('部分', 0.010027541146987453), ('放疗', 0.0099990988897575), ('腺癌', 0.009841534561588207), 
#     ('鳞癌', 0.009739382035478084), ('多', 0.00958973565787696), ('发育', 0.00956165510358892), 
#     ('淋巴结', 0.009533956609527014), ('停经', 0.00932700317409951), ('狭窄', 0.00927871945241034), 
#     ('损伤', 0.00925063055672048), ('级别', 0.009146166317375906), ('；', 0.009131120860735715), 
#     ('肝', 0.009118478691624472), ('增生', 0.009097910127369548), ('扩张', 0.009041297893163), 
#     ('可能', 0.009013613182341472), ('提示', 0.00890833763002825), ('妊娠', 0.0088026459167797), 
#     ('大', 0.0088026459167797), ('子宫', 0.008780858835617677), ('管状腺瘤', 0.00872420717273166), 
#     ('卵巢', 0.008696531739397361), ('肿瘤', 0.008375588427819937), ('坏死', 0.00829537900456825),
#      ('冠心病', 0.0082677173454356), ('特指', 0.008050596331305497), ('高', 0.007941330912261216), 
#      ('b超', 0.007941330912261216), ('形成', 0.007831584783774515), ('m0', 0.007831584783774515), 
#      ('原因', 0.00777713100518858), ('出血', 0.00777713100518858), ('畸形', 0.007721349516379488), 
#      ('直肠', 0.007638253665810212), ('上', 0.007527009030098604), ('胎盘', 0.007500553919678759), 
#      ('右', 0.007499376326431845), ('不良', 0.007499376326431845), ('右肺', 0.007472103100024968), 
#      ('组织', 0.00730296060316215), ('可能性', 0.007275337586304654), ('恶性肿瘤', 0.00716251905310134),
#      ('占位', 0.00710493686562461)]


co_exist = {}
def find_pairs(x, y):
    global co_exist
    for word in x:
        if word not in punc_set and word in word_tfidf.keys() and word_tfidf[word] > 0.001 and len(word) > 1:
            key = word + '-' + y
            if key in co_exist.keys():
                co_exist[key] += 1
            else:
                co_exist[key] = 1

def fill_co_exsit_dict(x_list, y_list):
    for x in tqdm(x_list):
        x_l = seg.cut(x)
        for y_l in y_list:
            for y in y_l:
                find_pairs(x_l, y)

fill_co_exsit_dict(x_list, y_list)

with open('co_exist_tfidf_lenth>1.pkl', 'wb') as f:
    pickle.dump(co_exist, f)


import pickle
with open('co_exist_tfidf_lenth>1.pkl', 'rb') as f:
    co_exist = pickle.load(f)
sortt = sorted(co_exist.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)