# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 15:06:51 2025

@author: ilyas
"""

import math
import random
import re
import codecs


class ngramLM:
    """Ngram Language Model Class"""
    
    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        self.allTokens = []
        # TO DO 

    def trainFromFile(self,fn):
        with codecs.open(fn, 'r', encoding='utf-8') as file:
            corpus = file.read().strip()

        token_pattern = r"""(?x)
        (?:[A-ZÇĞIİÖŞÜ]\.)+
        | \d+(?:\.\d*)?(?:\'\w+)?
        | \w+(?:-\w+)*(?:\'\w+)?
        | \.\.\.
        | [][,;.?():_!#^+$%&><|/{()=}\"\'\\\"\`-]
        """
        turkish_lowercase_table = str.maketrans(
            "Iİ",
            "ıi"
        )
        lines = corpus.split('\n')
        new_tokens = ["<s>"]
        unique_tokens = []

        j=0
        for line in lines:
            j+=1
            if line.strip() == '':
                continue

            tokens = re.findall(token_pattern, line)
            i=0
            for token in tokens:
                token = token.translate(turkish_lowercase_table) 
                token = token.lower()

                if token not in unique_tokens:
                    unique_tokens.append(token)

                i+=1
                if token != '.' and token != '?' and token != '!' and token != '\n':
                    new_tokens.append(token)
                else:
                    new_tokens.append(token)
                    new_tokens.append('</s>')
                    if i != len(tokens):
                        new_tokens.append('<s>')

            if tokens and tokens[-1] not in ['.', '?', '!']:
                new_tokens.append('</s>')
            if j != len(lines):
                new_tokens.append('<s>')
        
        self.allTokens = new_tokens
        unique_tokens.append('<s>')
        unique_tokens.append('</s>')
        self.numOfTokens = len(new_tokens)
        self.sizeOfVocab = len(unique_tokens)

        sentences = []
        sentence=[]
        for token in new_tokens:
            sentence.append(token)
            if token == '</s>':
                sentences.append(sentence)
                sentence = []

        self.sentences = sentences
        self.numOfSentences = len(sentences)

        # TO DO

    def vocab(self):
        token_dict = {}
        for token in self.allTokens:
            if token not in token_dict.keys():
                token_dict[token] = 1
            else:
                token_dict[token] += 1
        token_dict = dict(sorted(token_dict.items(), key=lambda item: (-item[1], item[0])))
        vocabList = [(key, value) for key, value in token_dict.items()]
        self.vocab = vocabList
        return vocabList


    def bigrams(self):
        bigram_dict = {}
        for sentence in self.sentences:
            i = 0
            while True:
                bigram_tuple = (sentence[i], sentence[i+1])
                if bigram_tuple not in bigram_dict.keys():
                    bigram_dict[bigram_tuple] = 1
                else:
                    bigram_dict[bigram_tuple] += 1
                i +=1
                if i+1 == len(sentence):
                    break

        bigram_dict = dict(sorted(bigram_dict.items(), key=lambda item: (-item[1], item[0])))
        bigramList = [(key, value) for key, value in bigram_dict.items()]
        return bigramList 
        

    def unigramCount(self, word):
        token_count = 0
        for token in self.allTokens:
            if token == word:
                token_count +=1
        return token_count


    def bigramCount(self, bigram):
        bigramList = self.bigrams()
        bigram_count = 0
        for item in bigramList:
            if bigram == item[0]:
                return item[1]
        return 0
        
        # TO DO

    def unigramProb(self, word):
        return self.unigramCount(word) / self.numOfTokens
 
    #def bigramProb(self, bigram):
        # TO DO
        # returns unsmoothed bigram probability value

    def unigramProb_SmoothingUNK(self, word):
        return (self.unigramCount(word) +1) / (self.numOfTokens + (self.sizeOfVocab+1))


lm = ngramLM()  
lm.trainFromFile("hw02_tinyCorpus.txt")
print(lm.bigrams())
print(lm.bigramCount(('bir','yer'))   )
print(lm.bigramCount(('yer','bir'))   )
print(lm.bigramCount(('yer','alır'))  )
print(lm.bigramCount(('bir','başlık')))
print(lm.bigramCount(('bir','kuş'))   )
print(lm.bigramCount(('yer','kuş'))   )
print(lm.bigramCount(('kuş','bir')))

    #def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        # returns smoothed bigram probability value

    #def sentenceProb(self,sent):
        # TO DO 
        # sent is a list of tokens
        # returns the probability of sent using smoothed bigram probability values

    #def generateSentence(self,sent=["<s>"],maxFollowWords=1,maxWordsInSent=20):

        
        # TO DO 
        # sent is a list of tokens
        # returns the generated sentence (a list of tokens)
