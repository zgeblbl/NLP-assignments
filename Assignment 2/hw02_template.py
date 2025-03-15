# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 15:06:51 2025

@author: ilyas
"""

import math
import random
import re
import codecs


# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""
    
    # Create Empty ngramLM
    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        # TO DO 

    # INSTANCE METHODS
    def trainFromFile(self,fn):
        # TO DO
    
    def vocab(self):
        # TO DO
  
    def bigrams(self):
        # TO DO

    def unigramCount(self, word):
        # TO DO

    def bigramCount(self, bigram):
        # TO DO

    def unigramProb(self, word):
        # TO DO
        # returns unsmoothed unigram probability value

    def bigramProb(self, bigram):
        # TO DO
        # returns unsmoothed bigram probability value

    def unigramProb_SmoothingUNK(self, word):
        # TO DO
        # returns smoothed unigram probability value

    def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        # returns smoothed bigram probability value

    def sentenceProb(self,sent):
        # TO DO 
        # sent is a list of tokens
        # returns the probability of sent using smoothed bigram probability values

    def generateSentence(self,sent=["<s>"],maxFollowWords=1,maxWordsInSent=20):
        # TO DO 
        # sent is a list of tokens
        # returns the generated sentence (a list of tokens)

            
            
        
        
