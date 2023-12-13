import os

import numpy as np

from common.np import *
from common.config import *
import pandas as pd

src_path = "../dataset/training/sourcedata_preprocess"

corpuses = np.load(os.path.join(src_path, "corpuses.npy"), allow_pickle=True)
id_to_word = np.load(os.path.join(src_path, "i2w.npy"), allow_pickle=True)
word_to_id = {}

for i in range(len(id_to_word)):
    word_to_id[id_to_word[i]] = i

corpus = list()
for cp in corpuses:
    cp = list(cp)
    corpus.extend(cp)
    corpus.append(word_to_id["@"])
    corpus.append(word_to_id["@"])
    corpus.append(word_to_id["@"])
    corpus.append(word_to_id["@"])
    corpus.append(word_to_id["@"])
corpus = np.array(corpus)

import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu

# 하이퍼파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 1

# 데이터 읽기
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# 모델 등 생성
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
if GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
