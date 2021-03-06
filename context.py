from helper import *
class Context :
    ''' Represents some definition of context to search for NE and rules(such as sentence)'''
    def __init__(self, tok_list) :
        self.tok_list = tok_list
        #Map from token to list of indicies
        self.word_map = build_tok_index(tok_list)

    def __getitem__(self, index) :
        return self.tok_list[index]

    def valid_index(self, index) :
        if index < 0 or index >= len(self.tok_list) :
            return False
        return True

    def match(self, sequence) :
        ''' takes list of tokens to match, outputs list of indicies denoting the
        first index of a sequence hit in the context's token list'''
        if len(sequence) == 0  :
            return []
        seq_len = len(sequence)
        spline = []

        #Match match all tokens in sequence.
        for tok in sequence :
            if tok not in self.tok_list :
                return []
            spline.append(self.word_map[tok])

        #Find sequence by comparing indicies (filtering)
        candidates = spline[0]
        for i in range(1, len(sequence)) :
            candidates = [j for j in candidates if j+i in spline[i]]
        return candidates
    '''
    #Rule generating functions
    def rule_windows(self, index, ne, fwd_window=1, rev_window=1) :
        r = Rule()
        if fwd_window > 0 :
            fwd_index = index+len(ne)+fwd_window
            if self.valid_index(fwd_index) :
                r.fwd_window = self.tok_list[index+len(ne):fwd_index]
            else :
                return None

        if rev_window > 0 :
            rev_index = index - rev_window
            if self.valid_index(rev_index) :
                r.rev_window = self.tok_list[rev_index:index]
            else :
                return None
        return r
    '''
    #NE generating functions
    def extract_ne(self, start_index, end_index) :
        ''' Given index bounds, return a list of NE between those two bounds '''
        if not valid_index(start_index) or not valid_index(end_index) :
            return []

        discovered_ne = []
        curr_ne = []
        recording = False
        for i in range(start_index, end_index) :
            tok = self.tok_list[i]
            if tok[0].isupper() :
                if recording == False :
                    recording = True
                curr_ne.append(tok)
            else :
                if recording == True :
                    recording = False
                    discovered_ne.append(tuple(curr_ne))
        if recording == True :
            discovered_ne.append(tuple(curr_ne))
        return discovered_ne

    def generate_rules(self, named_entity) :
        result = []
        matches = self.match(list(named_entity))
        '''Filter context based on further tokens in the entity
        name. The matches list will have no indicies if there are
        no hits for the entire NE'''
        for index in matches :
            #Can call various functions to populate rules
            r = self.rule_windows(index, named_entity)
            if r != None :
                result.append(r)
        return result
