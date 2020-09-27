import numpy as np
from scipy.optimize import linprog
from gensim.models import Word2Vec
from load_icd import load_icd, load_additional_icd, clean
import time
import pickle
import multiprocessing as mp
from tqdm import tqdm


def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    # print(result.status)
    # input()
    return result.fun


def word_mover_distance(x, y, lower_bound = False):
    """WMD（Word Mover's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    # x = np.array(x)
    # y = np.array(y)
    if lower_bound:
        lower_bound_distance = np.linalg.norm(np.mean(x, axis = 0) - np.mean(y, axis = 0))
        return lower_bound_distance
    else:
        p = np.ones(x.shape[0]) / x.shape[0]
        q = np.ones(y.shape[0]) / y.shape[0]
        D = np.sqrt(np.square(x[:, None] - y[None, :]).mean(axis=2))
        return wasserstein_distance(p, q, D)

def word_rotator_distance(x, y, lower_bound = False):
    """WRD（Word Rotator's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    # x = np.array(x)
    # y = np.array(y)
    if lower_bound:
        z_x = np.sum([np.linalg.norm(x[i]) for i in range(len(x))])
        z_y = np.sum([np.linalg.norm(y[i]) for i in range(len(y))])
        lower_bound_distance = np.linalg.norm(np.sum(x, axis = 0)/z_x - np.mean(y, axis = 0)/z_y)
        return lower_bound_distance
    else:
        x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
        y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
        p = x_norm[:, 0] / x_norm.sum()
        q = y_norm[:, 0] / y_norm.sum()
        D = 1 - np.dot(x / x_norm, (y / y_norm).T)
        return wasserstein_distance(p, q, D)

def load_w2v(file_path):
    W = Word2Vec.load(file_path)
    word_lst = ["[PAD]"] + W.wv.index2word + ["[UNK]"]
    word2id = {word:idx for idx, word in enumerate(word_lst)}
    id2word = {idx:word for word, idx in word2id.items()}
    vector = W.wv.vectors
    vector_mean = np.mean(vector, axis=0)
    vector_word = np.concatenate((np.zeros_like(vector_mean).reshape(1, -1), vector, vector_mean.reshape(1, -1)), axis=0)
    return vector_word, word2id, id2word

def get_text_emb(x, vectors, word2id):
    x_emb = []
    for char in x:
        if char in word2id.keys():
            x_emb.append(vectors[word2id[char]])
        else:
            x_emb.append(vectors[word2id["[UNK]"]])
    x_emb = np.array(x_emb)
    return x_emb


def find_distance_xy(x_list, output_y_num = 200, metrics = 'wmd'):
    icd2str, match_list = load_icd()
    str2standard = load_additional_icd()
    embedding_path = "/media/sdc/GanjinZero/jiangsu_info/word2vec_5_300.model"
    vectors, word2id, _ = load_w2v(embedding_path)
    output = []
    for x in tqdm(x_list):
        dist_dict = {}
        pred_y = []
        x_emb = get_text_emb(x, vectors, word2id)
        for y in match_list:
            y_emb = get_text_emb(y, vectors, word2id)
            if metrics == 'wmd':
                dist = word_mover_distance(x_emb, y_emb, lower_bound = True)
            elif metrics == 'wrd':
                dist = word_rotator_distance(x_emb, y_emb, lower_bound = True)
            dist_dict[y] = dist
        sorted_dist = sorted(dist_dict.items(), key = lambda kv:kv[1])
        dist_dict = {}
        for y in sorted_dist[:500]:
            y_emb = get_text_emb(y[0], vectors, word2id)
            if metrics == 'wmd':
                dist = word_mover_distance(x_emb, y_emb)
            elif metrics == 'wrd':
                dist = word_rotator_distance(x_emb, y_emb)
            dist_dict[y] = dist
        sorted_dist = sorted(dist_dict.items(), key = lambda kv:kv[1])
        for item in sorted_dist[:output_y_num]:
            pred_y.append(item[0])
        output.append(pred_y)
    return output

def find_distance_xy_mp(lock, id, q, x_list, output_y_num = 200, metrics = 'wmd'):
    icd2str, match_list = load_icd()
    str2standard = load_additional_icd()
    embedding_path = "/media/sdc/GanjinZero/jiangsu_info/word2vec_5_300.model"
    vectors, word2id, _ = load_w2v(embedding_path)
    output = []
    with open('./wrd_200/'+str(id)+'.pkl','wb') as f:
        for x in tqdm(x_list):
            dist_dict = {}
            pred_y = []
            x_emb = get_text_emb(x, vectors, word2id)
            for y in match_list:
                y_emb = get_text_emb(y, vectors, word2id)
                if metrics == 'wmd':
                    dist = word_mover_distance(x_emb, y_emb, lower_bound = True)
                elif metrics == 'wrd':
                    dist = word_rotator_distance(x_emb, y_emb, lower_bound = True)
                dist_dict[y] = dist
            sorted_dist = sorted(dist_dict.items(), key = lambda kv:kv[1])
            dist_dict = {}
            for y in sorted_dist[:400]:
                y_emb = get_text_emb(y[0], vectors, word2id)
                if metrics == 'wmd':
                    dist = word_mover_distance(x_emb, y_emb)
                elif metrics == 'wrd':
                    dist = word_rotator_distance(x_emb, y_emb)
                dist_dict[y[0]] = dist
            sorted_dist = sorted(dist_dict.items(), key = lambda kv:kv[1])
            for item in sorted_dist[:output_y_num]:
                pred_y.append(item[0])
            output.append({x:pred_y})
            pickle.dump({x:pred_y}, f)
    # lock.acquire()
    # q.put(output)
    # lock.release()

if __name__ =="__main__":
    with open("train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    x_list = []
    y_list = []
    for line in lines:
        l = line.strip().split("\t")
        x_list.append(clean(l[0]))

    q = mp.Queue()
    Lock = mp.Lock()
    # find_distance_xy_mp(11, q, x_list[:2], output_y_num = 200, metrics = 'wmd')
    processes = []
    for i in range(10):
        processes.append(mp.Process(target=find_distance_xy_mp,args=(Lock, i, q, x_list[i*800: (i+1)*800], 200, 'wrd')))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    # out = []
    # for k in range(20):
    #     out1 = q.get()
    #     with open('./wmd_200/wmd_200'+str(k)+'.pkl', 'wb') as f:
    #         pickle.dump(out1, f)
    #     out.append(out1)
    
    # with open('./wmd_200/wmd_200.pkl', 'wb') as f:
    #     pickle.dump(out, f)
    
# import pickle
# l = []
# with open('./wmd_200/2.pkl', 'rb') as f:
#     while True:
#         try:
#             l.append(pickle.load(f))
#         except EOFError:
#             break


    #time: 253.36724996566772
    # overall time after optimize: 9.6095
    #time: 0.553236722946167 100times
    #
