'''
--- Core logic used by chatbot_gui
--- This module loads the trained assets, converts user text to BoW
--- Predicts intent with a confidence threshold, and returns a response from intents.json.
'''

import json
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

##------ To ensure tockeniser is present at run time as well.-------##
try: 
    nltk.data.find('tockenizers/punkt')
except LookupError: 
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

#loading previously saved words, classes and the trained model.
#chatbot uses them to process user input and predict the intent
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))
model = load_model('models/chatbot_model.keras')
intents = json.loads(open('data/intents.json').read())


##----- Processing user input ------##

def clean_up_sentence(sentence):
    # tockenize + lemmatize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

  
def bag_of_words(sentence, words_vocab):
    # chatbot uses 'bag of words model to represent the user's input as a vector of 0s and 1s over training vocabulary'
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words_vocab]
    return np.array(bag)


##------- Predicting the intent -------##

def predict_class(senetence):
    """
    Predict intent probabilities and return sorted intents above threshold.
    Returns list of dicts: [{"intent": tag, "probability": str(prob)}]
    (most likely intent)
    """
    bow = bag_of_words(senetence,words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    #sort by strength of probability
    results.sort(key = lambda x:x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({ "intent":classes[r[0]], "probability":str(r[1])})
    return return_list


##------Generating a response --------##
def get_response(intents_list,intents_json):
    """
    Pick a random response from the top predicted intent.
    Fallback to 'noanswer' tag if nothing passes threshold.
    """
    if not intents_list:
    # try a defined 'noanswer' intent if present
        for intent in intents_json['intents']:
            if intent['tag'] == 'noanswer':
                return random.choice(intent['responses'])
        return "I'm not sure I understood that."

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I'm not sure I understood that."


##------ Public function used by the GUI -------##

def get_response_from_bot(user_message):
    """
    Public function used by the GUI.
    Cleans input, predicts class, and returns a response.
    """
    if not user_message or not user_message.strip():
        return "Please type something! :)"
    intents_list = predict_class(user_message)
    return get_response(intents_list, intents)


# ---- Optional quick CLI test ----
if __name__ == "__main__":
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            break
        print("Bot:", get_response_from_bot(msg))