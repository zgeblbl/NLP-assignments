"""
Assignment 1 - Code template for AIN442/BBM497

@author: İsmail Furkan Atasoy
"""
import regex as re
import codecs

def initialVocabulary():
    
    # You can use this function to create the initial vocabulary.
    
    return list("abcçdefgğhıijklmnoöprsştuüvyzwxq"+
                "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ"+
                "0123456789"+" "+
                "!'^#+$%&/{([)]=}*?\\_-<>|.:´,;`@€¨~\"é")

def bpeCorpus(corpus, maxMergeCount=10):     
    corpusList = re.split(r'\s+', corpus)
    vocab = initialVocabulary()
    tokenizedCorpus = []
    for word in corpusList:
        charList = [" "]
        for char in word:
            charList.append(char)
        charList.append("_")
        tokenizedCorpus.append(charList)

    mergeList = []
    j=0
    while j <maxMergeCount:
        pairDict = {}
        for wordList in tokenizedCorpus:
            for i in range(0, len(wordList)-1):
                pair = (wordList[i], wordList[i+1])
                if pair in pairDict:
                    pairDict[pair] +=1
                else:
                    pairDict[pair] = 1
        if(len(pairDict) == 0):
            break

        sortedPairDict = dict(sorted(pairDict.items(), key=lambda item: (-item[1], item[0])))
        mergeItem = next(iter(sortedPairDict.items()))
        vocab.append(mergeItem[0][0]+mergeItem[0][1])
        tempList = (mergeItem[0], mergeItem[1])
        mergeList.append(tempList)
        
        new_tokenized_corpus = []
        for wordList in tokenizedCorpus:
            merged_list = []
            i = 0
            while i < len(wordList) - 1:
                if (wordList[i], wordList[i + 1]) == mergeItem[0]:
                    merged_list.append(wordList[i] + wordList[i + 1])
                    i += 2
                else:
                    merged_list.append(wordList[i])
                    i += 1
            if i < len(wordList):
                merged_list.append(wordList[i])
            new_tokenized_corpus.append(merged_list)
        tokenizedCorpus = new_tokenized_corpus
        j +=1
    
    # TO DO
    # You can refer to Example 1, 2 and 3 for more details.
    # Should return (Merges, Vocabulary, TokenizedCorpus)
    return (mergeList, vocab, tokenizedCorpus)


def bpeFN(fileName, maxMergeCount=10):
    with codecs.open(fileName, 'r', encoding='utf-8') as file:
        corpus = file.read().strip()

    (Merges,Vocabulary,TokenizedCorpus)=bpeCorpus(corpus, maxMergeCount) 

    # TO DO
    # You can refer to Example 4 and 5 for more details.
    # Should return (Merges, Vocabulary, TokenizedCorpus)
    return (Merges,Vocabulary,TokenizedCorpus)

def bpeTokenize(str, merges):

    merged_list = tuple(pair[0][0] + pair[0][1] for pair in merges)
    words = re.split(r'\s+', str.strip())
    tokenList = []
    for word in words:
        tokenList.append([" "] + list(word) + ["_"])
    
    for merge in merges:
        mergeToken = merge[0][0]+merge[0][1]
        tokenizedStr = []
        for token in tokenList:
            merges = []
            i= 0
            while i<(len(token)-1):
                if (token[i]+token[i+1] == merge[0][0]+merge[0][1]):
                    merges.append(mergeToken)
                    i += 2
                else:
                    merges.append(token[i])
                    i += 1
            if i < len(token):
                merges.append(token[i])
            tokenizedStr.append(merges)

        tokenList = tokenizedStr


    # TO DO
    # You can refer to Example 6, 7 and 8 for more details.
    
    return tokenList


def bpeFNToFile(infn, maxMergeCount=10, outfn="output.txt"):
    
    # Please don't change this function. 
    # After completing all the functions above, call this function with the sample input "hw01_bilgisayar.txt".
    # The content of your output files must match the sample outputs exactly.
    # You can refer to "Example Output Files" section in the assignment document for more details.
    
    (Merges,Vocabulary,TokenizedCorpus)=bpeFN(infn, maxMergeCount)
    outfile = open(outfn,"w",encoding='utf-8')
    outfile.write("Merges:\n")
    outfile.write(str(Merges))
    outfile.write("\n\nVocabulary:\n")
    outfile.write(str(Vocabulary))
    outfile.write("\n\nTokenizedCorpus:\n")
    outfile.write(str(TokenizedCorpus))
    outfile.close()

bpeFNToFile("hw01_bilgisayar.txt",1000, "hw01-output1.txt")
bpeFNToFile("hw01_bilgisayar.txt",200, "hw01-output2.txt")
