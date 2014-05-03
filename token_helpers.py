from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
def sentence_tokenize(text) :
    tokenized_sentences = []
    sent_chunker = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_chunker.tokenize(text)
    for s in sentences : 
        sent_toks = [w for w in word_tokenize(s) if pass_filters(w)]
        tokenized_sentences.append(sent_toks)
    return tokenized_sentences

def pass_filters(tok) :
    filters = [lambda w : w in stopwords.words('english'),  #Ignore stopwords
                lambda w : len(w) == 0,                     #Empty Token
                lambda w : w == '``' or w == "''",          #Another form of noisy token
                lambda w : len(w) == 1 and not str.isalnum(w[0])] #Single punctuation tokens
    for test in filters :
        if test(tok) :
            return False
    return True


