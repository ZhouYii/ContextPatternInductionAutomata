from TFIDF import TFIDF
from collections import Counter
from token_helpers import *
from triggerword_impl import *
import ConfigParser
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
cfg = ConfigParser.ConfigParser()
cfg.read("config.ini")

def load_dict(file) :
    names = set()
    f = open(file, "r")
    for line in f :
        line = line.strip()
        if len(line) == 0 :
            continue
        names.add(tuple(word_tokenize(line)))
    return names

per_dict = load_dict(cfg.get("Dictionaries", "PER"))
#Extract all contexts for the PER type
corpus_dir = cfg.get("Global","CorpusDir")
file_list = [corpus_dir+'/'+f for f in listdir(corpus_dir)\
                            if isfile(join(corpus_dir,f))]
contexts = []
#Extract context from name list
for doc in file_list :
    new_contexts = extract_context(per_dict, doc, int(cfg.get("Global","WindowSz")))
    if new_contexts != None and len(new_contexts) > 0 :
        contexts.extend(new_contexts)

dominating_words = get_dominating_words(contexts, cfg.get("Global","CorpusDir"))
count = Counter(dominating_words)
trigger_words = count.most_common(int(cfg.get("Global","NumDominatingWords")))
trigger_words = [w[0] for w in trigger_words]
trimmed_dict = trim_contexts(contexts, trigger_words)
automata_dict = construct_automata(trimmed_dict)

