
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
model.load_params('../model/Rnnlm.pkl')

# start 문자와 skip 문자 설정
start_word = '저는'
start_id = word_to_id[start_word]
# 문장 생성
word_ids = model.generate(start_id)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

txt = txt.replace("@@", "\n\n")
txt = txt.replace("@", " ")
txt = txt.replace(" .", ".")
txt = txt.replace(" ,", ",")
txt = txt.replace(" ?", "?")
txt = txt.replace(" !", "!")
txt = txt.replace("  ", " ")
print(txt)
