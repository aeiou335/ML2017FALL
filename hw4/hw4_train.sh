#!/bin/bash
wget https://www.dropbox.com/s/wddllag0rn9yuym/nofunwithword2vec.bin
python3 train.py $1 $2 nofunwithword2vec.bin tokenizer.pickle