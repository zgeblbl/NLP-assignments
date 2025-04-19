# Özge Bülbül - 2220765008

from datasets import load_dataset
import re
import string
import nltk
import math
#nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score

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

class NaiveBayesClassifier:
    def __init__(self):
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.prior_pos = 0
        self.prior_neg = 0
        self.pos_counter = Counter()
        self.neg_counter = Counter()
        self.stop_words = set(stopwords.words("english"))

        # two lists added to store positive and negative words
        self.pos_list = []
        self.neg_list = []

    def fit(self, train_df):
        # split the training data into positive and negative
        pos_df = train_df[train_df['label'] == 1]
        neg_df = train_df[train_df['label'] == 0]

        # calculate prior probabilities
        self.prior_pos = len(pos_df) / len(train_df)
        self.prior_neg = len(neg_df) / len(train_df)

        pos_words = []
        neg_words = []

        # collect all words from positive and negative samples
        for _, row in pos_df.iterrows():
            pos_words.extend(row['text'].split())
        for _, row in neg_df.iterrows():
            neg_words.extend(row['text'].split())

        # total word counts
        self.total_pos_words = len(pos_words)
        self.total_neg_words = len(neg_words)

        # count word frequencies
        self.pos_counter = Counter(pos_words)
        self.neg_counter = Counter(neg_words)

        # build the vocab
        vocab = set()
        for text in train_df['text']:
            for word in text.split():
                vocab.add(word)
                
        self.vocab_size = len(vocab)

        # store all positive and negative words separately
        for text in pos_df['text']:
            self.pos_list.extend(text.split())
        for text in neg_df['text']:
            self.neg_list.extend(text.split())

        return train_df

    def predict(self, text):
        text = preprocess_text(text)    # preprocess the input text (if not preprocessed)

        # calculate initial log-priors
        log_prior_pos = math.log(self.prior_pos)
        log_prior_neg = math.log(self.prior_neg)

        log_pos_sum = log_prior_pos
        log_neg_sum = log_prior_neg

        # for each word calculate its contribution to the total log-probabilities
        for word in text.split():
            wi_count_pos = self.pos_counter[word]
            wi_count_neg = self.neg_counter[word]

            # laplace smoothing
            log_prob_word_given_pos = math.log((wi_count_pos + 1) / (self.total_pos_words + self.vocab_size))
            log_prob_word_given_neg = math.log((wi_count_neg + 1) / (self.total_neg_words + self.vocab_size))

            log_pos_sum += log_prob_word_given_pos
            log_neg_sum += log_prob_word_given_neg

        if log_pos_sum > log_neg_sum:
            y_predicted = 1
        else:
            y_predicted = 0

        return y_predicted, log_pos_sum, log_neg_sum
    

stop_words = set(stopwords.words("english"))

# preprocess the training and test texts
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

# create and train the Naive Bayes classifier
nb = NaiveBayesClassifier()
nb.fit(train_df)

# example code for testing the model
'''print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])

prediction1 = nb.predict(test_df.iloc[0]["text"])
prediction2 = nb.predict("This movie will be place at 1st in my favourite movies!")
prediction3 = nb.predict("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment.")
print(f"{"Positive" if prediction1[0] == 1 else "Negative"}")
print(prediction1)
print(f"{"Positive" if prediction2[0] == 1 else "Negative"}")
print(prediction2)
print(f"{"Positive" if prediction3[0] == 1 else "Negative"}")
print(prediction3)

print(preprocess_text("This movie will be place at 1st in my favourite movies!"))
print(preprocess_text("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment."))

y_true = test_df['label'].values
y_pred = [nb.predict(text)[0] for text in test_df['text']]
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")'''