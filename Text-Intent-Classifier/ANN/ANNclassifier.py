# using natural language toolkit
import nltk
import json
import os
import numpy as np
import time
import datetime
import tkinter as tk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.snowball import SnowballStemmer
#stemmer1 = SnowballStemmer("english")
# sigmoid function
sentence=''

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoidDerivative(output):
    return output*(1-output)

def wordsTokenizer(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bagOfWords(sentence, corpus_words, show_details=False):
    # tokenize the pattern
    sentence_words = wordsTokenizer(sentence)
    # bag of words
    bag = [0]*len(corpus_words)
    for s in sentence_words:
        for i,w in enumerate(corpus_words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def setSynapses(sentence, show_details=False):
    x = bagOfWords(sentence.lower(), corpus_words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bag of words:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

def train(X, y, hidden_neurons=10, alpha=0.1, epochs=50000, dropout=False, dropout_percent=0.5):

    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(y[0])) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(y[0]))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2


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
        layer_2_delta = layer_2_error * sigmoidDerivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoidDerivative(layer_1)

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
               'words': corpus_words,
               'intents': intents
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)


def intentClassifier(sentence, show_details=False):
    sentence = str(sentence)
    results = setSynapses(sentence, show_details)
    #print(results)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[intents[r[0]],r[1]] for r in results]
    #print ("%s \n classification: %s" % (sentence, return_results))
    return return_results





if __name__ == "__main__":
    # word stemmer
    stemmer = LancasterStemmer()

    # import chat-bot intents training file
    with open('intents.json') as json_data:
        sample_dataset = json.load(json_data)

    # capture unique stemmed words in the training corpus
    corpus_words = []
    intents = []
    training_data = []
    record_list =[]
    #ignore = ['?']

    for record in sample_dataset['intent_examples']:

        if record['intent'] not in intents:
            intents.append(record['intent'])
        record_list.append(record)

        for word in nltk.word_tokenize(record['text']):
            # ignore a few things
            if word not in ["?", "'s", ","]:
                # stem and lowercase each word
                if word == "'m":
                    word = "am"
                stemmed_word = stemmer.stem(word.lower())
                corpus_words.append(stemmed_word)
                #documents.append((stemmed_word,intent_examples['intent']))
                #class_words[intent_examples['intent']].extend([stemmed_word])

    for record in record_list:
        training_data.append((nltk.word_tokenize(record["text"]), record["intent"]))

    corpus_words = list(set(corpus_words))

    print (len(training_data), "dataset records")
    print (len(intents), "intents", intents)
    print (len(corpus_words), "unique stemmed words", corpus_words)

    # create our training data
    training_input = []
    training_output = []
    # create an empty array for our output
    output_empty = [0] * len(intents)

    # training set, bag of words for each sentence
    for record in training_data:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = record[0]
        #print(pattern_words)
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in corpus_words:
            bag.append(1) if w in pattern_words else bag.append(0)

        training_input.append(bag)
        # output is a '0' for each tag and '1' for current tag
        output = list(output_empty)
        output[intents.index(record[1])] = 1
        training_output.append(output)


    # sample training/output
    i = 10
    w = training_data[i][0]
    print ([stemmer.stem(word.lower()) for word in w])
    print (training_input[i])
    print (training_output[i])

    X = np.array(training_input)
    y = np.array(training_output)

    start_time = time.time()

    train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

    elapsed_time = time.time() - start_time
    print ("processing time:", elapsed_time, "seconds")

    # probability threshold
    ERROR_THRESHOLD = 0.2
    # load our calculated synapse values
    synapse_file = 'synapses.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])

    # classify("show me a mexicon place in the center")
    # classify("how are you today?")
    # classify("talk to you tomorrow, bye")
    # classify("search thai cuisine in city")
    # classify("get me some lunch")
    # print ()
    print(intentClassifier("looking for a dinner place near city center?", show_details=True))

    def onPressEnter(text):
        text = user_input.get()
        #print(text)
        #print(type(text))
        res.set(intentClassifier(text))
    def onButtonClick():
        onPressEnter(text)

    def clearAll():
        user_input.delete(0, 'end')
        intent_class.delete(0, 'end')

    row_offset = 1
    root = tk.Tk()
    root.title("ANN based Intent Classifier")
    root["padx"] = 30
    root["pady"] = 20
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    Label1 = tk.Label(root)
    Label1["text"] = "Enter a sentence :"
    Label1.grid(row=row_offset,padx=5,sticky=tk.E,column=1)

    text = tk.StringVar()
    user_input = tk.Entry(root, textvariable=text)
    user_input["width"] = 40
    user_input.grid(row=row_offset,sticky = tk.W, column=3)
    user_input.bind("<Return>", onPressEnter)
    user_input.focus_set()

    Label2 = tk.Label(root)
    Label2["text"] = "Intent Class :"
    Label2.grid(row=row_offset+2,padx=5,sticky=tk.E,column=1)

    res = tk.StringVar()
    intent_class = tk.Entry(root, textvariable=res)
    intent_class["width"] = 25
    intent_class.grid(row=row_offset+2,sticky=tk.W, column=3)

    classify_button = tk.Button(root, text="Classify", command=onButtonClick)
    classify_button.grid(row=5,column=4)

    reset_button = tk.Button(root, text="Reset", command=clearAll)
    reset_button.grid(row=5,column=2)

    root.mainloop()

    #
    # while(True):
    #     print()
    #     sentence = input("Enter a sentence: ")
    #     print(intentClassifier(sentence))
    #     print("If the intent is correct, do you want me to add this sentence to my taining dataset, (y/n)")
    #     if input("Press Y to try again: ") not in ['Y','y']:
    #         break
