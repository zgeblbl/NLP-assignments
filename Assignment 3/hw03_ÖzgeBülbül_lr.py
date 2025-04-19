# Özge Bülbül - 2220765008

from datasets import load_dataset
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
import math
#nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score

# load the IMDB dataset
dataset = load_dataset("imdb")

def preprocess_text(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()

    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(word)
        else:
            words.append(' ')
    text = " ".join(words)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def bias_scores(train_df):
    pos_df = train_df[train_df['label'] == 1]
    neg_df = train_df[train_df['label'] == 0]
    bias_scores_dict = {}
    fp_dict = {}
    fn_dict = {}
    
    # count word frequencies in positive and negative samples - store them in a dict
    for _, row in pos_df.iterrows():
        for word in row['text'].split():
            if word not in fp_dict:
                fp_dict[word] = 1
            else:
                fp_dict[word] += 1
    for _, row in neg_df.iterrows():
        for word in row['text'].split():
            if word not in fn_dict:
                fn_dict[word] = 1
            else:
                fn_dict[word] += 1
    
    # calculate bias score for each word
    for _, row in train_df.iterrows():
        for word in row['text'].split():
            if word not in bias_scores_dict.keys():
                bias_scores_dict[word] = abs((fp_dict.get(word, 0) - fn_dict.get(word, 0)) / (fp_dict.get(word, 0) + fn_dict.get(word, 0))) * math.log((fp_dict.get(word, 0) + fn_dict.get(word, 0)))
    
    # sort top 10,000 words by bias score
    sorted_score_dict = dict(sorted(bias_scores_dict.items(), key=lambda x: (-x[1], x[0]))[:10000])

    top_words = []
    for key in sorted_score_dict.keys():
        top_words.append((key, fp_dict.get(key,0), fn_dict.get(key,0), (fp_dict.get(key,0) + fn_dict.get(key,0)), sorted_score_dict.get(key,0)))

    return top_words


stop_words = set(stopwords.words("english"))

# preprocess train and test datasets
train_preprocessed = []
test_preprocessed = []
for text in dataset['train']['text']:
    train_preprocessed.append(preprocess_text(text))
for text in dataset['test']['text']:
    test_preprocessed.append((preprocess_text(text)))

dataset['train'] = dataset['train'].add_column('preprocessed', train_preprocessed)
dataset['test'] = dataset['test'].add_column('preprocessed', test_preprocessed)

train_df = pd.DataFrame({
    'text': dataset['train']['preprocessed'],
    'label': dataset['train']['label']
})
test_df = pd.DataFrame({
    'text': dataset['test']['preprocessed'],
    'label': dataset['test']['label']
})

# get top biased words
scores = bias_scores(train_df)

# extract top words
top_words_list = []
for item in scores:
    top_words_list.append(item[0])

vectorizer = CountVectorizer(vocabulary=top_words_list)

X_train = vectorizer.transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']

train_acc_list = []
test_acc_list = []

# train logistic regression models with max_iter from 1 to 25
for i in range(1, 26):
    lr_model = LogisticRegression(max_iter=i)
    lr_model.fit(X_train, y_train)

    train_pred = lr_model.predict(X_train)
    test_pred = lr_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 26), train_acc_list, label='Train', marker='o')
plt.plot(range(1, 26), test_acc_list, label='Test', marker='o')
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy Score')
plt.title('Accuracy vs Number of Iterations')
plt.legend()
plt.grid(True)
plt.show()

'''
MY ANALYSIS:
In train df, accuracy improves with more iterations.
In test df, accuracy improves with more iterations but stabilizes after 10 iterations.
I would prefer using a model with around 15 iterations since it achieves high accuracy without the unnecessary training time.
Choosing a model with more iterations wouldn't increase accuracy that much, however; it would make training longer.
''' 
