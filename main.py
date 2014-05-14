from TFIDF import TFIDF
from collections import Counter
from token_helpers import *
from triggerword_impl import *
import ConfigParser
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
from NE_Recognizer import NE_Recognizer
def get_names(dictionary) :
    l = list()
    for type in dictionary.keys() :
        l.extend(dictionary[type])
    return l

cfg = ConfigParser.ConfigParser()
cfg.read("config.ini")

#Dictionary of all name lists
name_lists = load_name_lists(cfg)
corpus_dir = cfg.get("Global","CorpusDir")
boundary_detector = NE_Recognizer(corpus_dir)
file_list = [corpus_dir+'/'+f for f in listdir(corpus_dir)\
                            if isfile(join(corpus_dir,f))]
print len(file_list)
pattern_scores = dict()

for step in range(0, int(cfg.get("Global","Iterations"))) :
    print "Iter #"+str(step)
    context_dict = init_dict()
    for doc in file_list :
        #Takes name dict as input, output name dict-mapping to context lists
        context_dict = extract_context(context_dict, name_lists, doc, int(cfg.get("Global","WindowSz")))

    #Take name dict, output words dict by type
    dominating_dict = get_dominating_words(context_dict, cfg.get("Global","CorpusDir"))
    trigger_dict = get_trigger_words(dominating_dict)
    #Trims the patterns by trigger word matches
    trimmed_dict = trim_contexts(context_dict, trigger_dict)
    automata_dict = construct_automata(trimmed_dict)

    #Extract using automata
    #Rules hits : type->[(match tokens, pattern)]
    rule_hits = automata_extractions(automata_dict, file_list, cfg.get("Global", "MaxMatchLen"))
    rule_hits = prune_by_pattern_score(rule_hits, pattern_scores)
    ne_dict, ne_to_pattern, punish = extract_entities(rule_hits, boundary_detector)
    # Vague labels are removed from this round
    for name in punish :
        ne_dict.pop(name, None)
    promoted_dict, punish = filter_promotion(ne_dict, name_lists)
    print "promoted:" + str(promoted_dict)
    pattern_scores = rank_patterns(pattern_scores, punish, \
                                    get_names(promoted_dict), ne_to_pattern)
    for k in promoted_dict.keys() :
        print set(promoted_dict[k])
        name_lists[k] = name_lists[k].union(set(promoted_dict[k]))
    f = open("output/Iter"+str(step),"w")
    for k in promoted_dict.keys() :
        f.write("**Category : "+str(k)+"\n")
        f.write("\n**Promoted:\n")
        f.write(str(k)+":"+str(promoted_dict[k]))

        f.write("\n**Trigger Patterns:\n")
        for i in trimmed_dict[k].items() :
            f.write(str(i[0])+":"+str(i[1])+'\n')

        f.write("\n**Rule Hits:\n")
        for i in rule_hits[k] :
            f.write(str(i)+'\n')
            '''
        f.write("\n**All Extracted Names:\n")
        for i in ne_dict[k] :
            f.write(str(i)+'\n')
            '''
    f.close()
    f = open("output/NameList"+str(step), "w")
    for k in name_lists.keys() :
        l = to_names(name_lists[k])
        for name in l :
            f.write(str(name) + '\t' + str(k) + '\n')
    f.close()
    if len(get_names(promoted_dict)) == 0 :
        break
