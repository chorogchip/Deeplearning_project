

import os
import json
from common.np import *
import pandas as pd

src_path = "../dataset/training/sourcedata"
dest_path = "../dataset/training/sourcedata_preprocess"
file_list = os.listdir(src_path)
print("open %d files" % len(file_list))

src_path_2 = "../dataset/validation/sourcedata"
dest_path_2 = "../dataset/validation/sourcedata_preprocess"
file_list_2 = os.listdir(src_path_2)
print("open %d files" % len(file_list_2))

file_names = list()

for file in file_list:
    file_names.append(src_path + '/' + file)
file_names.append("@@@")
for file in file_list_2:
    file_names.append(src_path_2 + '/' + file)

i = 0
word_to_id = {}
id_to_word = list()
corpuses = list()

for file in file_names:
    if file == "@@@":
        np.save(os.path.join(dest_path, "corpuses_train"), np.array(corpuses, dtype=object), allow_pickle=True)
        i = 0

        continue

    with open(file, 'r', encoding='utf-8') as fdata:
        data = json.load(fdata)
        txt = str(data['essay_txt'])

        txt = txt.replace("#@문장구분#", " @ ")
        txt = txt.replace("\n\n", " @@ ")
        txt = txt.replace("\n", " @@ ")

        txt2 = ""
        for ch in txt:
            if ".,<>?/\'\";:[]{}()-_=+!#$%^&*~`1234567890".find(ch) >= 0 :
                txt2 += " " + ch + " "
            else:
                txt2 += ch

        # 1문단으로 구성된 데이터는 사용하지 않습니다.
        cnt = txt2.count("@@")
        if cnt == 0:
            continue

        spl = txt2.split(" ")
        spl2 = [i for i in spl if i != ""]

        for word in spl2:
            if word not in word_to_id:
                new_id = len(id_to_word)
                id_to_word.append(word)
                word_to_id[word] = new_id

        lst = [word_to_id[w] for w in spl2]
        corpuses.append(np.array(lst))

        i = i + 1
        if i % 1000 == 0:
            print("processed %d files" % i)

np.save(os.path.join(dest_path, "corpuses_validation"), np.array(corpuses, dtype=object), allow_pickle=True)
np.save(os.path.join(dest_path, "i2w"), np.array(id_to_word), allow_pickle=True)
print("totally processed %d files, word count : %d" % (i, len(word_to_id)))
