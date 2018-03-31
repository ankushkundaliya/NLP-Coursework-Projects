import os
import numpy as np
import pandas as pd
import nltk
import time
import datetime
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import json

stop_words = set(stopwords.words("english"))
stemmer = LancasterStemmer()
traindir = "/home/mtech20/Documents/Machine Learning/ling-spam/train-mails"
testdir = "/home/mtech20/Documents/Machine Learning/ling-spam/test-mails"

class ANNclassifier:
    confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
    dictionary = {}
    total_wordcount = 0
    wordsList = []
    featureVectors = []
    similarity = []
    category = []
    spam_dict = {}
    ham_dict = {}
    X = []
    y = []

    def make_Dictionary(self, traindir):
        for f in os.listdir(traindir):
            email = os.path.join(traindir,f)
            with open(email) as m:
                data = m.read()
                words = nltk.word_tokenize(data)
                if 'sp' in f:
                    self.category.append('spam')
                    self.count_words(words, self.spam_dict)
                else:
                    self.category.append('ham')
                    self.count_words(words, self.ham_dict)

    def count_words(self, words, counts_dict):
        for word in words:
            if word not in stop_words and word.isalpha() and len(word) > 1:
                word = stemmer.stem(word.lower())
                self.total_wordcount += 1
                if word in counts_dict:
                    counts_dict[word] += 1
                else:
                    counts_dict[word] = 1

    def merge_Dictionary(self):
        for k,v in self.ham_dict.items():
            self.dictionary.setdefault(k,[]).append(v)
            self.dictionary.setdefault(k,[]).append(0)
        for k,v in self.spam_dict.items():
            if k in self.dictionary.keys():
                self.dictionary[k][1] = v
            else:
                self.dictionary.setdefault(k,[]).append(0)
                self.dictionary.setdefault(k,[]).append(v)
        list_to_remove = list(self.dictionary)
        #print(len(self.dictionary))
        for k in list_to_remove:
            if abs(self.dictionary[k][0] - self.dictionary[k][1]) < 100:
                del self.dictionary[k]
        print(len(self.dictionary))


    def createfeatureVectors(self, traindir):
        for f in os.listdir(traindir):
            classVector = [-1, -1]
            email = os.path.join(traindir,f)
            if 'sp' in f:
                classVector = [0,1]
            else:
                classVector = [1,0]
            with open(email) as m:
                data = m.read()
                words = nltk.word_tokenize(data)
                vector = []
                wordList = []
                for word in words:
                    if word not in stop_words and word.isalpha() and len(word) > 1:
                        word = stemmer.stem(word.lower())
                        wordList.append(word)
                for word in self.dictionary.keys():
                    if word in wordList:
                        vector.append(1)
                    else:
                        vector.append(0)
                vect = [vector, classVector]
                #print(len(vector))
                self.featureVectors.append(vect)
        #X = []
        #y = []
        for v in self.featureVectors:
            self.X.append(v[0])
            self.y.append(v[1])
        #print(self.y)

    #compute sigmoid nonlinearity
    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    #convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

    def test_Classifier(self, testdir):
        # probability threshold
        ERROR_THRESHOLD = 0.03
        # load our calculated synapse values
        synapse_file = 'synapses.json'
        with open(synapse_file) as data_file:
            synapse = json.load(data_file)
            synapse_0 = np.asarray(synapse['synapse0'])
            synapse_1 = np.asarray(synapse['synapse1'])

        confusion_matrix = {"TP":0, "FP":0, "FN":0, "TN":0}

        for f in os.listdir(testdir):
            classVector = [-1, -1]
            SPAM = False
            HAM = False
            email = os.path.join(testdir,f)
            if 'sp' in f:
                classVector = [0,1]
                SPAM = True
            else:
                classVector = [1,0]
                HAM = True
            with open(email) as m:
                data = m.read()
                words = nltk.word_tokenize(data)
                vector = []
                wordList = []
                for word in words:
                    if word not in stop_words and word.isalpha() and len(word) > 1:
                        word = stemmer.stem(word.lower())
                        wordList.append(word)
                for word in self.dictionary.keys():
                    if word in wordList:
                        vector.append(1)
                    else:
                        vector.append(0)
                vect = [vector, classVector]
            #print ("msg:", sentence, "\n bow:", x)
            # input layer is our bag of words
            l0 = vector
            # matrix multiplication of input and hidden layer
            l1 = self.sigmoid(np.dot(l0, synapse_0))
            # output layer
            l2 = self.sigmoid(np.dot(l1, synapse_1))
            classes = ["ham", "spam"]
            prediction = [[i,r] for i,r in enumerate(l2) if r>ERROR_THRESHOLD ]
            prediction.sort(key=lambda x: x[1], reverse=True)
            #print(prediction)
            return_results =[[classes[r[0]],r[1]] for r in prediction]
            #print ("%s \n classification: %s" % (lines, return_results))
            #print(return_results)
            if(return_results[0][0])=="spam":
                if (SPAM == True):
                    confusion_matrix['TP']+=1
                else:
                    confusion_matrix['FP']+=1
            else:
                if(SPAM == True):
                    confusion_matrix['FN']+=1
                else:
                    confusion_matrix['TN']+=1
        print(confusion_matrix)
        SP = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])

        SR = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])

        A = (confusion_matrix['TP'] + confusion_matrix['TN']) / (confusion_matrix['TP'] + confusion_matrix['TN'] + confusion_matrix['FP'] + confusion_matrix['FN'])

        print("Precision :")
        print(SP)
        print("Recall :")
        print(SR)
        print("Accuracy :")
        print(A)




    def train(self, hidden_neurons=10, alpha=0.1, epochs=50000, dropout=False, dropout_percent=0.5):

        #print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
        #print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
        np.random.seed(1)
        #print(type(len(self.X[0])))
        self.X = np.array(self.X)
        #print(self.X)
        self.y = np.array(self.y)
        last_mean_error = 1
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((len(self.X[0]), hidden_neurons)) - 1
        synapse_1 = 2*np.random.random((hidden_neurons,2)) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)

        for j in iter(range(epochs+1)):

            # Feed forward through layers 0, 1, and 2
            layer_0 = self.X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))

            if(dropout):
                layer_1 *= np.random.binomial([np.ones((len(self.X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

            layer_2 = self.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = self.y - layer_2

            if (j% 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                    break

                # in what direction is the target value?
                # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * self.sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)
            #print(layer_1.T)
            #print(layer_0)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if(j > 0):
                synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'classes': ['Ham', 'Spam']
              }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)
            print ("saved synapses to:", synapse_file)


    def classify(sentence, show_details=False):
        results = think(sentence, show_details)
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results =[[classes[r[0]],r[1]] for r in results]
        #print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

def main():
    ANNspamfilter = ANNclassifier()
    ANNspamfilter.make_Dictionary(traindir)
    ANNspamfilter.merge_Dictionary()
    ANNspamfilter.createfeatureVectors(traindir)
    start_time = time.time()
    ANNspamfilter.train()
    elapsed_time = time.time() - start_time
    print ("processing time:", elapsed_time, "seconds")

    ANNspamfilter.test_Classifier(testdir)

if __name__ == '__main__':
    main()
