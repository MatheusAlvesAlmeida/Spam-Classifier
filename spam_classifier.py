from sklearn.pipeline import Pipeline;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import classification_report;
from sklearn.naive_bayes import MultinomialNB;
from sklearn.feature_extraction.text import TfidfTransformer;
from sklearn.feature_extraction.text import CountVectorizer;
from nltk.corpus import stopwords;
from IPython.display import Image;
import nltk;
import pandas as pd;
import string;


mensagens = [line.rstrip() for line in open(
    'smsspamcollection/SMSSpamCollection')];
mensagens = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t', names=["tipo", "mensagem"])


def preProcessamento(msg):
    noPont = [char for char in msg if char not in string.punctuation]
    noPont = ''.join(noPont)
    return [word for word in noPont.split() if word.lower() not in stopwords.words('english')]

# Isso pode demorar um pouco
bow_transformer = CountVectorizer(analyzer=preProcessamento).fit(mensagens['mensagem'])

messages_bow = bow_transformer.transform(mensagens['mensagem'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

msg_train, msg_test, tipo_train, tipo_test = \
train_test_split(mensagens['mensagem'], mensagens['tipo'], test_size=0.2)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=preProcessamento)),  # Tokeniza as mensagens
    ('tfidf', TfidfTransformer()),  # Faz a transformação em TF-IDF
    ('classifier', MultinomialNB()),  # Define a classe que realizará nossa classificação.
])

pipeline.fit(msg_train,tipo_train)

predictions = pipeline.predict(msg_test)

print(classification_report(predictions,tipo_test))