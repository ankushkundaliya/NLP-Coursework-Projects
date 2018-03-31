**Text Intent Classifier for a Restaurant Search Chatbot**

  The inherent problem of pattern-based heuristics chatbot model is that patterns should be programmed manually, and it is not an easy task, especially if the chatbot has to correctly distinguish hundreds of intents. 
  Imagine that you are building a customer service bot and the bot should respond to a refund request. Users can express it in hundreds of different ways: “I want a refund”, “Refund my money”, “I  need my money back”. At the same time, the botshould respond differently if the same words are used in another context: “Can I request a refund if I don’t like the service?”, “What is your refund policy?”. 
  Humans are not good at writing patterns and rules for natural language understanding, computers are much better at this task. Machine learning lets us train an intent classification algorithm. We just need a training set of a few hundred or thousands of examples, and it will pick up patterns in the data.

Python version :  Python 3.6.1 

Packages used : 
  1. nltk 3.2.2 
  2. numpy 1.12.1 
  3. tkinter 8.6 
  4. json 2.0.9 

**How to run the application? **
 
Step 1 : Double click the IntentClassifierApp.py file or run in cmd using command- python IntentClassifierApp.py 
 
Step 2 : Once the training data is pre-processed and our ANN model is built, a GUI based application window will open. 
 
Step 3 : Enter a sentence in the text input field.

( Text intent can be anyone of the 4 intents: “Greeting”, “Restaurant Search”, “Affirm” or “Goodbye”. 
  Text with other intents may be miss-classified.) 
 
Step 4 : Hit enter or click “Classify” button to get the intent of entered text. 
 
Step 5 : Reset button is used to reset the text input fields. 
