
import os
from common.np import *

src_path = "../dataset/training/sourcedata_preprocess"

corpuses_t = np.load(os.path.join(src_path, "corpuses_train.npy"), allow_pickle=True)
corpuses_v = np.load(os.path.join(src_path, "corpuses_validation.npy"), allow_pickle=True)
id_to_word = np.load(os.path.join(src_path, "i2w.npy"), allow_pickle=True)
word_to_id = {}

for i in range(len(id_to_word)):
    word_to_id[id_to_word[i]] = i

corpus_t = list()
for cp in corpuses_t:
    cp = list(cp)
    corpus_t.extend(cp)
    corpus_t.append(word_to_id["@"])
    corpus_t.append(word_to_id["@"])
    corpus_t.append(word_to_id["@"])
    corpus_t.append(word_to_id["@"])
    corpus_t.append(word_to_id["@"])
corpus_t = np.array(corpus_t)

corpus_v = list()
for cp in corpuses_v:
    cp = list(cp)
    corpus_v.extend(cp)
    corpus_v.append(word_to_id["@"])
    corpus_v.append(word_to_id["@"])
    corpus_v.append(word_to_id["@"])
    corpus_v.append(word_to_id["@"])
    corpus_v.append(word_to_id["@"])
corpus_v = np.array(corpus_v)


# coding: utf-8
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen


corpus = corpus_v
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
model.load_params('../model_new/Rnnlm_new.pkl')

import random
from common.util import *
import pickle

pkl_file = '../embedding/cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs_e = params['word_vecs']
    word_to_id_e = params['word_to_id']
    id_to_word_e = params['id_to_word']


cos_sim_dist_sq_val = 0
div_sum = 0

for i in range(100):

    start_id = random.randrange(0, len(id_to_word))
    start_word = id_to_word[start_id]
    # 문장 생성
    word_ids = model.generate(start_id)

    for i1 in range(len(word_ids)):
        word1 = id_to_word[word_ids[i1]]
        vec1 = word_vecs_e[word_to_id_e.get(word1)]
        for i2 in range(len(word_ids)):
            word2 = id_to_word[word_ids[i2]]
            vec2 = word_vecs_e[word_to_id_e.get(word2)]

            if len(vec1.shape) != 1 or len(vec2.shape) != 1:
                continue

            dist = i1 - i2
            dist *= dist
            cos_sim_dist_sq_val += cos_similarity(vec1, vec2) * dist
            div_sum += dist

print(cos_sim_dist_sq_val / div_sum)