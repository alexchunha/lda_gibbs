#!/usr/bin/env python

import numpy as np, termcolor as tc
from Lda import TopicWordPair as TWPair
from Lda import TopicWordMatrix as TWMatrix

class LdaModel:

    # Only supporting two colors for now, so the print function below only
    # distinguishes topid == 0 from topid > 0
    colors = ['blue', 'yellow']
    
    def __init__(self, n_topics, documents):

        self.T = n_topics
        self.D = len(documents)
        
        self.corpus = [[TWPair(word, d) for word in doc]
                       for d, doc in enumerate(documents)]
        
        self.dictionary = {word: idx for idx, word in
                           enumerate(
                               set(
                                   sum(documents, [])
                               )
                           )}
        
        self.matrix = TWMatrix(n_topics, self.corpus, self.dictionary)
        self.matrix.A = 1 / self.matrix.T  # 1 / 2
        self.matrix.B = 1 / self.matrix.D  # 1 / 3
    
    def sweep(self, n_passes = 1):
        self.matrix.sweep(n_passes)
    
    def get_document_token_topics(self, doc_id = None):
        if (doc_id is not None):
            return [token.t for token in self.corpus[doc_id]]
        else:
            return [[token.t for token in doc] for doc in self.corpus]
    
    def print_document_topics_symbol(self, symbol = b'\xe2\x80\xa2'.decode(), inplace = False):
        colored_docs = [''.join(tc.colored(symbol, self.colors[min(t, 1)])
                                for t in doc)
                        for doc in self.get_document_token_topics()]
        print(*colored_docs, sep = tc.colored('|', 'red'), end = ('\r' if inplace else '\n'))
        # Note that this will print all topid >= 1 as the same color
    
    def print_document_topics_tokens(self, inplace = False):
        colored_docs = [' '.join(tc.colored(token.word, self.colors[min(token.t, 1)])
                                 for token in doc)
                        for doc in self.corpus]
        print(*colored_docs, sep = ' | ', end = ('\r' if inplace else '\n'))
    
    def get_ddist(self):
        return np.array([self.matrix.get_document_topic_dist(d) for d in range(self.D)])
    
    def get_tdist(self):
        return [self.matrix.get_topic_word_dist(t) for t in range(self.T)]
    
    def get_word_topic_counts(self, word):
        counts = {t: 0 for t in range(self.T)}
        for doc in self.corpus:
            for token in doc:
                if word == token.word:
                    counts[token.t] += 1
        return counts
