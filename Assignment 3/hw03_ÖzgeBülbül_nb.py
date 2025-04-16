from datasets import load_dataset
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd

dataset = load_dataset("imdb")
#print(dataset["train"][0])

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(word)
    text = " ".join(words)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

stop_words = set(stopwords.words("english"))

train_preprocessed = []
test_preprocessed = []
for text in dataset['train']['text']:
    train_preprocessed.append(preprocess_text(text))
for text in dataset['test']['text']:
    test_preprocessed.append((preprocess_text(text)))

dataset['train'] = dataset['train'].add_column('preprocessed', train_preprocessed)
dataset['test'] = dataset['test'].add_column('preprocessed', test_preprocessed)

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

    def fit(self, train_df):
        df_size = len(train_df)
        pos_df = train_df[train_df['label'] == 1]
        neg_df = train_df[train_df['label'] == 0]

        self.prior_pos = len(pos_df) / df_size
        self.prior_neg = len(neg_df) / df_size

        pos_words = []
        neg_words = []

        for _, row in pos_df.iterrows():
            pos_words.extend(row['text'])
        for _, row in neg_df.iterrows():
            neg_words.extend(row['text'])

        self.total_pos_words = len(pos_words)
        self.total_neg_words = len(neg_words)

        vocab = []
        for words in train_df['text']:
            for word in words.split():
                if word not in vocab:
                    vocab.append(word)

        self.vocab_size = len(vocab)
        #print(self.total_pos_words)

        return train_df

    def predict(self, text):


        return text
    
train_df = pd.DataFrame({
    'text': dataset['train']['preprocessed'],
    'label': dataset['train']['label']
})

nb = NaiveBayesClassifier()
nb.fit(train_df)
print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])
