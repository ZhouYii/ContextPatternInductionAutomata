import nltk
from collections import Counter
from nltk import word_tokenize
from os import listdir
import math
from token_helpers import *
from os.path import isfile, join
from nltk.corpus import stopwords
lemma = nltk.WordNetLemmatizer()

def tokenize_doc(filepath) :
    f = open(filepath, "r")
    toks = word_tokenize(f.read())
    toks = [w.lower().replace('.','') for w in toks if pass_filters(w)]
    toks = [lemma.lemmatize(t) for t in toks]
    return toks

class TFIDF : 
    def count_freq(self, term, val) :
        ''' Update term frequency '''
        if self.term_freq.has_key(term) :
            self.term_freq[term] += val
        else :
            self.term_freq[term] = val

    def count_doc(self, term) :
        ''' Update document frequency '''
        if self.term_docnum.has_key(term) :
            self.term_docnum[term] += 1
        else :
            self.term_docnum[term] = 1

    def __init__(self, corpus_dir) :
        self.term_docnum = dict()
        self.term_freq = dict()

        files = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir,f))]
        self.num_docs = len(files)
        for f in files :
            toks = tokenize_doc(corpus_dir+'/'+f)
            freq_dict = Counter(toks)
            for item in freq_dict.items() :
                # ITEM : (Key, Freq)
                self.count_freq(item[0], item[1])
                self.count_doc(item[0])
        self.ordered_term_frequency = sorted(self.term_freq.items(), \
                key=lambda x : x[1], reverse = True)
        self.max_freq = self.ordered_term_frequency[0][1]

    def tf_idf(self, term) :
        ''' Calculate tf-idf for a term, based on training code '''
        if not (self.term_docnum.has_key(term) and self.term_freq.has_key(term)) :
            ''' If frequency is zero, the TF/IDF is also zero (divide by zero for
            IDF)'''
            return 0
        tf = 0.5 + float(0.5*self.term_freq[term]) / float(self.max_freq)
        idf = math.log(self.num_docs/float(self.term_docnum[term]),2)
        return tf*idf

    def idf(self,term) :
        ''' Calculate just the idf '''
        term = term.lower()
        if not self.term_docnum.has_key(term) :
            return 0
        return math.log(self.num_docs/float(self.term_docnum[term]),2)


