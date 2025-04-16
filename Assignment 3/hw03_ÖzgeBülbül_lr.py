from datasets import load_dataset
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

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