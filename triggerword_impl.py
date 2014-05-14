from token_helpers import *
import ConfigParser
from TFIDF import TFIDF
from Automata import Automata
import nltk
from collections import Counter
from nltk.corpus import stopwords
categories = ["PER", "LOC", "ORG"]
cfg = ConfigParser.ConfigParser()
cfg.read("config.ini")

def init_dict(fxn=list) :
    d = dict()
    for t in categories :
        d[t] = fxn()
    return d

def load_dict(file) :
    names = set()
    f = open(file, "r")
    for line in f :
        line = line.strip()
        if len(line) == 0 :
            continue
        names.add(tuple(word_tokenize(line)))
    return names

def load_name_lists(cfg) :
    name_lists = dict()
    for c in categories :
            name_lists[c] = load_dict(cfg.get("Dictionaries", c))
    return name_lists

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
        fwd_window = seq[skip_name_indx:min(len(seq),skip_name_indx+2)]
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

def extract_context(context_dict,name_dict, file, window) :
    types = name_dict.keys()
    f = open(file, "r")
    all_contexts = sentence_tokenize(f.read())
    for c in all_contexts :
        index = build_index(c)
        for t in types :
            name_list = name_dict[t]
            for name in name_list :
                hits = match_sequence(c,name,index)
                if len(hits) == 0 :
                    continue
                context_dict[t].extend(mine_context(c, hits, name, window))
    return context_dict

def get_dominating_words(context_dict, corpusdir) :
    tfidf = TFIDF(corpusdir)
    dominating = init_dict()
    cache = dict()
    for t in context_dict.keys() :
        contexts = context_dict[t]
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
                dominating[t].append(curr_max[0])
    return dominating

def trim_contexts(context_dict, trigger_dict) :
    ''' Assumes the context is reversed if needed, if the rules being inducted
    on are right-facing rules '''
    trimmed_dict = init_dict(dict)
    for t in context_dict.keys() :
        triggers = trigger_dict[t]
        context_list = context_dict[t]
        for tok in triggers :
            trimmed_dict[t][tok] = []
        for c in context_list :
            while len(c) > 0:
                tok = c[0]
                if tok in triggers :
                    if c.count('-ENT-') < 1 :
                        break
                    subseq = c[:min(len(c), c.index("-ENT-")+2)]
                    trimmed_dict[t][tok].append(subseq)
                    break
                c.pop(0)
    return trimmed_dict

def construct_automata(trigger_dict) :
    '''trigger_dict : type => triggers => rules '''
    automata_dict = init_dict(dict)
    for k in trigger_dict.keys() :
        trigger_to_context = trigger_dict[k]
        for trigger in trigger_to_context.keys() :
            automata_dict[k][trigger] = Automata(trigger)
            for contexts in trigger_to_context[trigger] :
                automata_dict[k][trigger].learn(contexts)
    return automata_dict

def automata_extractions(automata_dict, file_list, max_match_len) :
    extractions = init_dict()
    for filepath in file_list :
        sent_tok = sentence_tokenize(open(filepath,"r").read())
        for sentence in sent_tok :
            while len(sentence) > 0 :
                tok = sentence[0]
                for category in automata_dict.keys() :
                    if automata_dict[category].has_key(tok) :
                        automata = automata_dict[category][tok]
                        out = automata.match_context(sentence)
                        if len(out) < 2 or len(out) > max_match_len or \
                                len(out[0]) < 1 or len(out[1]) < 1 :
                            continue
                        match = tuple(out[0])
                        pattern =  out[1]
                        extractions[category].append((match, tuple(pattern)))
                if len(sentence) > 0 :
                        sentence.pop(0)
    return extractions

def extract_entities(rulematch_dict, boundary_detector) :
    def merge_dicts(into, outof) :
        penalize_names = []
        for k in outof.keys() :
            if not into.has_key(k) :
                into[k] = outof[k]
            else :
                into[k].extend(outof[k])
                penalize_names.append(k)
        return into, set(penalize_names)

    result, all_name_patterns, punish_names = dict(), dict(), set()
    for category in rulematch_dict.keys() :
        #names: name => extraction score
        names, singles, name_to_pattern = extract_entities_(rulematch_dict[category],boundary_detector)
        all_name_patterns, negative_names = merge_dicts(all_name_patterns, name_to_pattern)
        punish_names = punish_names.union(negative_names)
        #if len(singles) > 0 :
        #    result[category] = singles
        names = sorted(names.items(), key=lambda x: x[1], reverse=True)
        names = [n[0] for n in names if n[1] > int(cfg.get("Global","MinFreq"))-1]
        result[category] = names
    return result, all_name_patterns, punish_names

def extract_entities_(rule_hits, boundary_detector) :
    ''' Returns all the name entities and their scores '''
    names = dict()
    name_patterns = dict()
    singles_set = set()
    for r in rule_hits :
        seq, score, pattern = r[0][0], 1, r[1]#r[0][1], r[1]
        name_dict, singles = boundary_detector.extract_names(seq)
        #singles_set = singles_set.union(singles)
        for ne in singles :
            name_dict[ne] = 1
        for i in name_dict.items() :
            new_name = i[0]
            if not names.has_key(new_name) : 
                names[new_name] = 0
            names[new_name] += float(score)
            if not name_patterns.has_key(new_name) :
                name_patterns[new_name] = []
            name_patterns[new_name].append(tuple(pattern))
    return names, singles_set, name_patterns

def get_trigger_words(dominating_dict) :
    triggers = dict()
    for t in dominating_dict.keys() :
        dominating_words = dominating_dict[t]
        count = Counter(dominating_words)
        trigger_words = count.most_common(int(cfg.get("Global","NumDominatingWords")))
        triggers[t] = [w[0] for w in trigger_words \
                if w[0].lower() not in stopwords.words('english')]
    return triggers

def filter_promotion(ne_dict, name_lists) :
    multi_types = set()
    for category in ne_dict.keys() :
        names = to_names_order_preserving(ne_dict[category])
        name_set = set(names)
        ''' If cross check/redundant names '''
        for type in name_lists :
            multi_types = multi_types.union(name_set.intersection(name_lists[type]))
            #print "names:"+str(names)
            #print "name_list:"+str(name_lists[type])
            names = [n for n in names if n not in name_lists[type]]
        ne_dict[category] = names[:int(cfg.get("Global","NumToPromote"))]
    return ne_dict, multi_types

def to_names(extraction_list) :
    names = set()
    for e in extraction_list :
        if type(e) is str :
            names.add(e)
        else :
            names.add(" ".join(list(e)))
    return names

def to_names_order_preserving(extraction_list) :
    names = list()
    for e in extraction_list :
        if type(e) is str :
            names.append(e)
        else :
            names.append(" ".join(list(e)))
    return names

def prune_by_pattern_score(rule_hits, pattern_dict) :
    for k in rule_hits.keys() :
        for hit in rule_hits[k] :
            pattern = hit[1]
            if not pattern_dict.has_key(pattern) :
                pattern_dict[pattern] = 0
            if pattern_dict[pattern] < 0 and rule_hits[k].count(hit) > 0 :
                rule_hits[k].remove(hit)
    return rule_hits

def rank_patterns(pattern_scores, punish_ne, promote_ne, ne_patterns) :
    punish_patterns = [ne_patterns[ne] for ne in punish_ne if ne_patterns.has_key(ne)]
    promote_patterns = [ne_patterns[ne] for ne in promote_ne if ne_patterns.has_key(ne)]
    for p_list in punish_patterns :
        for p in p_list :
            p = tuple(p)
            pattern_scores[p] = -1
    for p_list in promote_patterns :
        for p in p_list :
            if pattern_scores.has_key(p) :
                pattern_scores[p] += 1
            else :
                pattern_scores[p] = 1
    return pattern_scores

