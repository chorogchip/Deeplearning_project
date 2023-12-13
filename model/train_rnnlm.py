# coding: utf-8
import os

import numpy as np

from common.np import *
from common.config import *
import pandas as pd

import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from rnnlm import Rnnlm


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


# 하이퍼파라미터 설정
batch_size = 35
wordvec_size = 100
hidden_size = 100  # RNN의 은닉 상태 벡터의 원소 수
time_size = 35     # RNN을 펼치는 크기
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 학습 데이터 읽기
#corpus, word_to_id, id_to_word = ptb.load_data('train')
#corpus_test, _, _ = ptb.load_data('test')
corpus = corpus_t
corpus_test = corpus_v
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 모델 생성
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 기울기 클리핑을 적용하여 학습
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
            eval_interval=20)
trainer.plot(ylim=(0, 500))

# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 퍼플렉서티: ', ppl_test)

# 매개변수 저장
model.save_params()
