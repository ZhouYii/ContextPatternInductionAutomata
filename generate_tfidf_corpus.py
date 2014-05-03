import nltk
import string
import os
from BeautifulSoup import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

path = 'test'
token_dict = {}

def tokenize(text):
    text = unicode(text).encode('utf-8', 'ignore')
    tokens = nltk.word_tokenize(text)
    return tokens

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        print no_punctuation
        token_dict[file] = no_punctuation

#this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
vals = []
for v in token_dict.values(): 
    try :
        response = tfidf.fit_transform([v])
        vals.append(v)
    except :
        pass
#tfidf.fit_transform(vals)
str = 'obama football this sentence has unseen text such as computer but also king lord juliet'
response = tfidf.transform([str])
feature_names = tfidf.get_feature_names()
print feature_names
for col in response.nonzero()[1]:
        print feature_names[col], ' - ', response[0, col]
