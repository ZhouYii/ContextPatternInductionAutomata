class Node :
    def __init__(self, tok) :
        self.tok = tok
        self.transition = dict()
        self.total = 0

    def count_transition(self, start, end) :
        if start != self.tok :
            return
        self.total += 1
        if not self.transition.has_key(end) :
            self.transition[end] = 1
        else :
            self.transition[end] += 1

    def decrement_transition(self, endpt) :
        if self.transition.has_key(endpt) :
            self.transition[endpt] -= 1
            if self.transition[endpt] < 1 :
                self.transition.pop(endpt, None)

    def get_transition_confidence(self, end_tok) :
        if not self.transition.has_key(end_tok) or not self.total > 0 :
            return 0
        return float(self.transition[end_tok])/float(self.total)

    def neighbors(self) :
        return self.transition.keys()

class Automata :
    id = 0
    def __init__(self, trigger) :
        self.trigger = trigger
        self.nodes = dict()
        self.nodes[trigger] = Node(trigger)

    def learn(self, seq) :
        if len(seq) == 0 or seq[0] != self.trigger :
            return
        for i in range(0, len(seq)-1) :
            node = self.get_node(seq[i])
            node.count_transition(seq[i], seq[i+1])
        last = seq[len(seq)-1]
        if not self.nodes.has_key(last) :
            self.nodes[last] = Node(last)
            self.nodes[last].total += 1

    def get_node(self, node_tok) :
        if not self.nodes.has_key(node_tok) :
            self.nodes[node_tok] = Node(node_tok)
        return self.nodes[node_tok]

    def neg_score(self, seq) :
        ''' Take a sequence representing a pattern and applies negative score to
        it '''
        if len(seq) == 0 or seq[0] != self.trigger :
            return
        for i in range(0, len(seq)-1) :
            if not self.nodes.has_key(seq[i]) :
                break
            node = self.nodes[seq[i]]
            node.decrement_transition(seq[i+1])

    def match_context(self, seq) :
        ''' Return a candidate tokens for -ENT- if possible '''
        #print "ROOT:"+str(self.trigger) +" SEQ:"+str(seq)
        if len(seq) == 0 or seq[0] != self.trigger :
            return tuple()
        confidence = 1.0
        seq.pop(0)
        node = self.nodes[self.trigger]
        n = node.neighbors()
        pattern = [node.tok]
        while len(seq) > 0 :
            tok = seq[0]
            if n.count('-ENT-') > 0 :
                n = self.nodes['-ENT-'].neighbors()
                end, prob = self.broad_match(seq, n)
                if end == -1 or prob <= 0:
                    return tuple()
                confidence *= prob
                pattern.extend(['-ENT-',seq[end]])
                return [[seq[:end], confidence], pattern]

            #Seq does not fit automata
            if not self.nodes.has_key(tok) :
                return tuple()

            if tok in n :
                # Follow Path
                confidence *= node.get_transition_confidence(tok)
                node = self.nodes[tok]
                pattern.append(tok)
                n = node.neighbors()
                seq.pop(0)
            else :
                return tuple()

        return tuple()

    def broad_match(self, seq, candidates) :
        # r-token, index, transition prob.
        curr_max = (None, -1, 0)
        for c in candidates :
            if seq.count(c) < 1 :
                continue
            index = seq.index(c)
            if index > curr_max[1] :
                transition_prob = self.nodes['-ENT-'].get_transition_confidence(c)
                curr_max = (c, index, transition_prob)
        return curr_max[1], curr_max[2]

