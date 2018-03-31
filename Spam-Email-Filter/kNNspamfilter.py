import os
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

stop_words = set(stopwords.words("english"))
stemmer = LancasterStemmer()
traindir = "/home/mtech20/Documents/Machine Learning/ling-spam/train-mails"
testdir = "/home/mtech20/Documents/Machine Learning/ling-spam/test-mails"

class kNNclassifier:
    total_wordcount = 0
    total_hamcount = 0
    total_spamcount = 0
    confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
    dictionary = {}
    featureVectors = []
    similarity = []
    category = []

    def makeDictionary(self, traindir):
        for f in os.listdir(traindir):
            email = os.path.join(traindir,f)
            with open(email) as m:
                if 'sp' in f:
                    self.category.append('spam')
                else:
                    self.category.append('ham')
                data = m.read()
                words = nltk.word_tokenize(data)
                for word in words:
                    if word not in stop_words and word.isalpha() and len(word) > 1:
                        word = stemmer.stem(word.lower())
                        if word not in self.dictionary:
                            self.dictionary[word] = False

    def createfeatureVectors(self, traindir):
        for f in os.listdir(traindir):
            email = os.path.join(traindir,f)
            with open(email) as m:
                data = m.read()
                words = nltk.word_tokenize(data)
                vector = []
                wordList = []
                for word in words:
                    if word not in stop_words and word.isalpha() and len(word) > 1:
                        word = stemmer.stem(word.lower())
                        wordList.append(word)
                for word in self.dictionary:
                    if word in wordList:
                        vector.append(1)
                    else:
                        vector.append(0)
                self.featureVectors.append(vector)

    def testClassifier(self, testdir):
        self.confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
        for f in os.listdir(testdir):
            email = os.path.join(testdir,f)
            spam = False
            ham = False
            with open(email) as m:
                data = m.read()
                vector = []
                wordList = []
                if 'sp' in f:
                    spam = True
                else:
                    ham = True
                words = nltk.word_tokenize(data)
                for word in words:
                    if word not in stop_words:
                        if word.isalpha() == True and len(word) >1:
                            word = stemmer.stem(word.lower())
                            wordList.append(word)
                for word in self.dictionary:
                    if word in wordList:
                        vector.append(1)
                    else:
                        vector.append(0)
                self.similarity = self.cosineSimilarity(vector)
                prediction = self.classify()
                if prediction == 'ham':
                    if ham == True:
                        self.confusion_matrix['TN'] += 1
                    elif spma == True:
                        self.confusion_matrix['FN'] += 1
                else:
                    if ham == True:
                        self.confusion_matrix['FP'] += 1
                    elif spam == True:
                        self.confusion_matrix['TP'] += 1
        return self.confusion_matrix

    def classify(self, k=5):
        neighbours = [-1, -2, -3, -4, -5]
        pos = [-1, -1, -1, -1, -1]
        spamCount = 0
        hamCount = 0
        i = 0

        for dist in self.similarity:
            if dist > neighbours[4]:
                neighbours[4] = dist
                pos[4] = i
            elif dist > neighbours[3]:
                neighbours[3] = dist
                pos[3] = i
            elif dist > neighbours[2]:
                neighbours[2] = dist
                pos[2] = i
            elif dist > neighbours[1]:
                neighbours[1] = dist
                pos[1] = i
            elif dist > neighbours[0]:
                neighbours[0] = dist
                pos[0] = i
            i += 1

        for p in pos:
            if self.category[p] == 'spam':
                spamCount += 1
            elif self.category[p]  == 'ham':
                hamCount += 1
        if spamCount > hamCount:
            return 'spam'
        else :
            return 'ham'

    def cosineSimilarity(self, v1):
        similarity = []
        for v2 in self.featureVectors:
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            similarity.append(dot_product / (norm_v1 * norm_v2))
        return similarity

kNNSpamFilter = kNNclassifier()
kNNSpamFilter.makeDictionary(traindir)
kNNSpamFilter.createfeatureVectors(traindir)
confusion_mat = kNNSpamFilter.testClassifier(testdir)
print(confusion_mat)

SP = confusion_mat['TP'] / (confusion_mat['TP'] + confusion_mat['FP'])
SR = confusion_mat['TP'] / (confusion_mat['TP'] + confusion_mat['FN'])
A = (confusion_mat['TP'] + confusion_mat['TN']) / (confusion_mat['TP'] + confusion_mat['TN'] + confusion_mat['FP'] + confusion_mat['FN'])

print("Precision :")
print(SP)
print("Recall :")
print(SR)
print("Accuracy :")
print(A)
