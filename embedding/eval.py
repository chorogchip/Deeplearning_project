# coding: utf-8
import sys
sys.path.append('..')
from common.util import most_similar, analogy
import pickle


pkl_file = 'cbow_params.pkl'
# pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# 가장 비슷한(most similar) 단어 뽑기
querys = ['나는', '그래서', '사람', '주제']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# 유추(analogy) 작업
print('-'*50)
analogy('왕', '남자', '여왕',  word_to_id, id_to_word, word_vecs)
analogy('하다', '했다', '가다',  word_to_id, id_to_word, word_vecs)
analogy('차', '차들', '어린이',  word_to_id, id_to_word, word_vecs)
analogy('좋다', '최고', '나쁘다',  word_to_id, id_to_word, word_vecs)
