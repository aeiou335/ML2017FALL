#!/bin/bash
wget https://www.dropbox.com/s/av8kwvm7zyrdpma/ver2.3_GRU.h5
wget https://www.dropbox.com/s/jbvgspcsmywa9ee/ver2.4_GRU.h5
wget https://www.dropbox.com/s/wddllag0rn9yuym/nofunwithword2vec.bin
python3 combine.py $1 $2 ver2.3_GRU.h5 ver2.4_GRU.h5 tokenizer.pickle