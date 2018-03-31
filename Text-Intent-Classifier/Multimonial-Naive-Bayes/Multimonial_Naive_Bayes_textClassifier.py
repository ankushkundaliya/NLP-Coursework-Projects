
# coding: utf-8

# In[143]:

# using natural language toolkit
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.snowball import SnowballStemmer
# word stemmer
stemmer = LancasterStemmer()
#stemmer1 = SnowballStemmer("english")


# In[144]:

# import chat-bot intents training file
with open('intents.json') as json_data:
    training_data = json.load(json_data)


# In[145]:

# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}

for c in training_data['intent_examples']:
     class_words[c['intent']] = []

for intent_examples in training_data['intent_examples']:
    for word in nltk.word_tokenize(intent_examples['text']):
        # ignore a few things
        if word not in ["?", "'s"]:
            # stem and lowercase each word
            if word == "'m":
                word = "am"
            stemmed_word = stemmer.stem(word.lower())
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            class_words[intent_examples['intent']].extend([stemmed_word])

# the number of occurances of the word in training corpus (word frequency)
#print ("Corpus words and counts: %s" % corpus_words)
# all words in each intent-class
#print ("Class words: %s" % class_words)


# In[146]:

# calculate a score for a given class taking into account word commonality
def calculate_class_score_commonality(sentence, class_name, show_details=True):
    score = 0
    for word in nltk.word_tokenize(sentence):
        word = stemmer.stem(word.lower())
        if word in class_words[class_name]:
            score += (1 / corpus_words[word])
            if show_details:
                print ("   match: %s (%s)" % (word, 1 / corpus_words[word]))
    return score


# In[147]:

# return the class with highest score for sentence
def find_intent(sentence):
    high_class = None
    high_score = 0
    for c in class_words.keys():
        #% (c, calculate_class_score_commonality(sentence, c)))
        score = calculate_class_score_commonality(sentence, c)
        print("Class: %s  Score: %s \n" % (c, score))
        if score > high_score:
            high_class = c
            high_score = score
    return high_class, high_score


# In[148]:

while(True):
    sentence = input("Enter a sentence to find its intent: ")
    print(find_intent(sentence))
    print("Press 'Y' to try again!")
    if input() not in ['Y', 'y']:
        break;
