import os
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

stop_words = set(stopwords.words("english"))
stemmer = LancasterStemmer()
traindir = "ling-spam/train-mails"
testdir = "ling-spam/test-mails"

class NBclassifier:

    total_wordcount = 0
    total_hamcount = 0
    total_spamcount = 0
    spam_prob = 0
    ham_prob = 0
    prior_spam = 0
    prior_ham = 0
    confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
    dictionary = {}
    likelihood = {}
    spam_dict = {}
    ham_dict = {}

    def test_Classifier(self):
        self.mails_actual_label_count = [0, 0]
        self.mails_predicted_label_count = [0, 0]
        self.confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}

        for f in os.listdir(testdir):
            email = os.path.join(testdir,f)
            words_corpus = []
            spam = False
            ham = False
            with open(email) as m:
                ham_total_prob = 1
                spam_total_prob = 1
                data = m.read()
                if 'sp' in f:
                    spam = True
                else:
                    ham = True
                words = nltk.word_tokenize(data)
                for word in words:
                    if word not in stop_words:
                        if word.isalpha() == True and len(word) >1:
                            word = stemmer.stem(word.lower())
                            if word in self.dictionary.keys():
                                pr_of_word_given_ham = self.dictionary[word][0] / self.total_hamcount
                                pr_of_word_given_spam = self.dictionary[word][1] / self.total_spamcount
                                ham_total_prob *= pr_of_word_given_ham
                                spam_total_prob *= pr_of_word_given_spam
                ham_total_prob *= self.prior_ham
                spam_total_prob *= self.prior_spam

                if(ham_total_prob > spam_total_prob):
                    self.mails_predicted_label_count[1] += 1
                    if ham == True:
                        self.confusion_matrix['TN'] += 1
                    elif spam == True:
                        self.confusion_matrix['FN'] += 1
                else:
                    self.mails_predicted_label_count[0] += 1
                    if ham == True:
                        self.confusion_matrix['FP'] += 1
                    elif spam == True:
                        self.confusion_matrix['TP'] += 1

        return self.confusion_matrix

    def count_words(self, words, counts_dict):
        for word in words:
            if word not in stop_words:
                if word.isalpha() == True and len(word) > 1:
                    word = stemmer.stem(word.lower())
                    self.total_wordcount += 1
                    if word in counts_dict:
                        counts_dict[word] += 1
                    else:
                        counts_dict[word] = 1

    def make_Dictionary(self, traindir):
        for f in os.listdir(traindir):
            email = os.path.join(traindir,f)
            with open(email) as m:
                data = m.read()
                words = nltk.word_tokenize(data)
                if 'sp' in f:
                    self.count_words(words,self.spam_dict)
                else:
                    self.count_words(words,self.ham_dict)

    def merge_Dictionary(self):
        for k,v in self.ham_dict.items():
            self.dictionary.setdefault(k,[]).append(v+1)
            self.dictionary.setdefault(k,[]).append(1)
        for k,v in self.spam_dict.items():
            if k in self.dictionary.keys():
                self.dictionary[k][1] = v+1
            else:
                self.dictionary.setdefault(k,[]).append(1)
                self.dictionary.setdefault(k,[]).append(v+1)
        list_to_remove = list(self.dictionary)
        for k in list_to_remove:
            if abs(self.dictionary[k][0] - self.dictionary[k][1]) < 100:
                del self.dictionary[k]

    def calc_probabilities(self):
        for k in self.dictionary.keys():
            self.total_spamcount += self.dictionary[k][1]
            self.total_hamcount += self.dictionary[k][0]
            count = self.dictionary[k][0] + self.dictionary[k][1]
            total_count = self.total_wordcount + 2*len(self.dictionary)
            self.likelihood[k] = count / total_count

        self.prior_spam = self.total_spamcount / total_count
        self.prior_ham = self.total_hamcount / total_count


NBspamfilter = NBclassifier()
NBspamfilter.make_Dictionary(traindir)
NBspamfilter.merge_Dictionary()
NBspamfilter.calc_probabilities()
confusion_mat = NBspamfilter.test_Classifier()
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
