from token_helpers import *
from TFIDF import TFIDF
import nltk
def match_sequence(toks, name, index) :
    ''' Match name in a sequence of tokens '''
    results = []
    tok_matches = [[] for i in range(0, len(name))]
    for i in range(0, len(name)):
        if not index.has_key(name[i]) :
            return []
        tok_matches[i] = index[name[i]]
    for i in tok_matches[0] :
        valid = True
        for j in range(1,len(name)) :
            if (i+j) not in tok_matches[j] :
                valid = False
                break
        if valid :
            results.append(i)
    return results

def mine_context(seq, hits, name, window) :
    contexts = []
    for i in hits :
        prev_window = seq[max(0, i-window):i]
        skip_name_indx = min(len(seq)-1, i+len(name))
        fwd_window = seq[skip_name_indx:min(len(seq),skip_name_indx+window)]
        prev_window.append('-ENT-')
        prev_window.extend(fwd_window)
        contexts.append(prev_window)
    return contexts

def build_index(seq) :
    index = dict()
    for i in range(0, len(seq)):
        if not index.has_key(seq[i]) :
            index[seq[i]] = []
        index[seq[i]].append(i)
    return index

def extract_context(names, file, window) :
    results = []
    f = open(file, "r")
    all_contexts = sentence_tokenize(f.read())
    for c in all_contexts :
        index = build_index(c)
        for name in names :
            hits = match_sequence(c,name,index)
            if len(hits) == 0 :
                continue
            results.extend(mine_context(c, hits, name, window))
    return results

def get_dominating_words(contexts, corpusdir) :
    tfidf = TFIDF(corpusdir)
    dominating = []
    cache = dict()
    for c in contexts :
        curr_max = (None, -1)
        for tok in c :
            if tok == "-ENT-" :
                break
            if not cache.has_key(tok) :
                cache[tok] = tfidf.idf(tok)
            if cache[tok] > curr_max[1] :
                curr_max = (tok, cache[tok])
        if curr_max[0] != None :
            dominating.append(curr_max[0])
    return dominating

def trim_contexts(context_list, triggers) :
    ''' Assumes the context is reversed if needed, if the rules being inducted
    on are right-facing rules '''
    trig_dict = dict()
    for t in triggers :
        trig_dict[t] = []
    for c in context_list :
        while len(c) > 0:
            tok = c[0]
            if tok in triggers :
                if c.count('-ENT-') < 1 :
                    break
                subseq = c[:min(len(c), c.index("-ENT-")+2)]
                trig_dict[tok].append(subseq)
                break
            c.pop(0)
    return trig_dict

def construct_automata(trigger_dict) :
    automata_dict = dict()
    for trigger in trigger_dict.keys() :
        automata_dict[trigger] = Automata(trigger)
        for contexts in trigger_dict[trigger] :
            Automata.learn(contexts)
    return automata_dict
