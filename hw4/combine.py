import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
print('---read data---')

with open(sys.argv[1], 'r', encoding = 'utf-8') as f:
    test_sentences = []
    s2 = []
    f.readline()
    for l_test in f:
        l_test = l_test.split(',', 1)
        test_sentences.append(l_test[1])

print('---tokenizer---')
with open(sys.argv[5], 'rb') as handle:
    token = pickle.load(handle)
test_seq = token.texts_to_sequences(test_sentences)
x_test = pad_sequences(test_seq, maxlen = 30)

model1 = load_model(sys.argv[3])
model2 = load_model(sys.argv[4])

print('---predict data---')

res = model1.predict(x_test)
res2 = model2.predict(x_test)

with open(sys.argv[2], "w") as f:
    f.write("id,label\n")
    for id in range(len(res)):
        ans = np.argmax(res[id] + res2[id])
        f.write("{},{}\n".format(id, ans))
        