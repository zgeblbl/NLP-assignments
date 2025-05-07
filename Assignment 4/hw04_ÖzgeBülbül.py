# Özge BÜLBÜL 2220765008

import gensim.downloader
import random
import numpy as np
model = gensim.downloader.load("word2vec-google-news-300")

def replace_with_similar(sentence, indices):
    most_similar_dict = {}
    new_sentence = ""
    tokens = []
    tokens = sentence.split()
    for i in indices:
        word = tokens[i]
        similars_tuple_list = model.most_similar(word, topn=5)
        most_similar_dict[word] = similars_tuple_list
    j = 0
    for i in range(len(tokens)):
        j += 1
        if i not in indices:
            new_sentence += tokens[i]
        else:
            chosen_tuple = random.choice(most_similar_dict[tokens[i]])
            chosen_word = chosen_tuple[0]
            new_sentence += chosen_word
        if j != len(tokens):
            new_sentence += " "
        
    return new_sentence, most_similar_dict

def sentence_vector(sentence):
    vector_dict = {}
    sentence_vec = []
    tokens = []
    tokens = sentence.split()
    for word in tokens:
        if word in model:
            vector = model[word]
        else:
            vector = np.zeros(300)
        vector_dict[word] = vector

    vectors = []
    for vec in vector_dict.values():
        vectors.append(vec)
    vectors_array = np.array(vectors)
    sentence_vec = vectors_array.mean(axis=0)

    return vector_dict, sentence_vec

def most_similar_sentences(file_path, query):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        sentences = [line.strip() for line in lines]
    
    results = []
    vector_dict_query, sentence_vec_query = sentence_vector(query)
    sim_list = []

    for sentence in sentences:
        vector_dict, sentence_vec = sentence_vector(sentence)
        similarity_score = np.dot(sentence_vec_query, sentence_vec) / (np.linalg.norm(sentence_vec_query) * np.linalg.norm(sentence_vec))
        sim_list.append((sentence, similarity_score))

    results = sorted(sim_list, key=lambda x: x[1], reverse=True)

    return results


# sentence = "I love AIN442 and BBM497 courses"
# indices = [1, 5]
# new_sentence, most_similar_dict = replace_with_similar(sentence, indices)
# print(most_similar_dict.keys(), end="\n\n")
# print(most_similar_dict["love"], end="\n\n")
# print(most_similar_dict["courses"], end="\n\n")
# print(new_sentence)
# -------------------------------------------------------------------------
# vector_dict, sentence_vec = sentence_vector("This is a test sentence")
# print(vector_dict.keys(), end="\n\n")
# print(vector_dict["This"][:5], end="\n\n")
# print(vector_dict["a"][:5], end="\n\n")
# print(len(vector_dict["test"]))

# query_vec = sentence_vector("Which courses have you taken at Hacettepe University ?")[1]
# sentence_vec = sentence_vector("Students have the chance to gain practice with the homework given in lab classes of universities .")[1]
# print(query_vec[:5])
# print(sentence_vec[:5])
# -------------------------------------------------------------------------
# file_path = "sentences.txt"
# query = "Which courses have you taken at Hacettepe University ?"
# results = most_similar_sentences(file_path, query)
# for sentence, score in results[:3]:
#     print(f"{score:.5f} -> {sentence}")
# -------------------------------------------------------------------------
# with open("output.txt", "w", encoding="utf-8") as f:
#     sentence = "NLP is a fascinating field of study and I love learning about it"
#     indices = [3, 4, 10]
#     new_sentence, most_similar_dict = replace_with_similar(sentence, indices)
    
#     print(most_similar_dict.keys(), end="\n\n", file=f)
#     print(most_similar_dict["fascinating"], end="\n\n", file=f)
#     print(most_similar_dict["field"], end="\n\n", file=f)
#     print(most_similar_dict["learning"], end="\n\n", file=f)
#     print(new_sentence, end="\n\n", file=f)
    
#     print("----------------------------------------------------------------------------------------------------", end="\n\n", file=f)

#     vector_dict, sentence_vec = sentence_vector("I am a student studying NLP at Hacettepe University")
#     print(vector_dict.keys(), end="\n\n", file=f)
#     print(vector_dict["I"][:5], end="\n\n", file=f)
#     print(vector_dict["studying"][145:150], end="\n\n", file=f)
#     print(vector_dict["Hacettepe"][295:], end="\n\n", file=f)

#     print("----------------------------------------------------------------------------------------------------", end="\n\n", file=f)

#     file_path = "sentences.txt"

#     query1 = "Is swimming a good sport ?"
#     results1 = most_similar_sentences(file_path, query1)
#     for sentence, score in results1[:3]:
#         print(f"{score:.5f} -> {sentence}", end="\n\n", file=f)

#     print("--------------------------------------------------", end="\n\n", file=f)

#     query2 = "Does Turkey have good universities ?"
#     results2 = most_similar_sentences(file_path, query2)
#     for sentence, score in results2[:3]:
#         print(f"{score:.5f} -> {sentence}", end="\n\n", file=f)

#     print("--------------------------------------------------", end="\n\n", file=f)

#     query3 = "What happened to your backpack ?"
#     results3 = most_similar_sentences(file_path, query3)
#     for sentence, score in results3[:3]:
#         print(f"{score:.5f} -> {sentence}", end="\n\n", file=f)