f = open("extracted","r")
o = open("rule_hits","w")
for l in f :
    l = l.strip()
    if len(l) <= 0 :
        continue
    tok_seq = l.split("###")
    s = " ".join(tok_seq)
    o.write(s+'\n')
