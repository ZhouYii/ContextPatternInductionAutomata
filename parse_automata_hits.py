from document import Document
from helper import *
from os import listdir
import pickle
import re
from NE_Recognizer import NE_Recognizer
from nltk import word_tokenize
from os.path import isfile, join

def merge_dict(big, small) :
    for k in small.keys() :
        if not big.has_key(k) :
            big[k] = small[k]
        else :
            big[k] += small[k]
    return big
filepath="test"
ne_r = NE_Recognizer()
totals = dict()
files = [ filepath+'/'+f for f in listdir(filepath) if isfile(join(filepath,f)) ]
for f in files :
    ne_r.train_document(Document(f))

f = open("extracted","r")
for l in f :
    l = l.strip()
    if len(l) <= 0 :
        continue
    tok_seq = l.split("###")
    name_dict = ne_r.extract_names(tok_seq)
    print name_dict
    totals = merge_dict(totals, name_dict)

names = sorted(totals.items(), key=lambda x:x[1], reverse=True)
for n in names :
    print n
