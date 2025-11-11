import numpy as np
import json
import random
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD 


##------ NLTK setup for first time download-------##
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
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

##-------- Load & Normalize training data --------##
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data/intents.json').read())

words = [] #vocabulary tockens
classes = [] #intent tags
documents = [] #(tocken_list, tag)
ignore_tockens = ['?','!','.',',',"'s"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tockenize
        tocken_list = nltk.word_tokenize(pattern) 
        words.extend(tocken_list)
        #add documents in the corpus
        # documents = combination between patterns and intents
        documents.append((tocken_list, intent['tag']))
        #add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

##-------- Lemmatize + lower words + filter punctuations ---------##
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_tockens]
words = sorted(set(words))
classes = sorted(set(classes))

# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

#saving vocabulary and classes
pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

#create tarining data(bag of words + one-hot labels)
training = []
#create empty array for output
output_empty = [0] * len(classes)
#training set, bag of words for each sentence
for doc in documents:
    # list of tokenized words for the pattern
    pattern_tockens = doc[0]
    tag = doc[1]
    #Lemmatize pattern tockens
    pattern_tockens = [lemmatizer.lemmatize(w.lower()) for w in pattern_tockens]

    #create our bag of words array with 1, if word match found in current pattern
    bag = [1 if w in pattern_tockens else 0 for w in words]

    #convert tag to one-hot vector
    #output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = output_empty[:]
    output_row[classes.index(tag)] = 1

    training.append([bag, output_row])

#shuffle the features for stability, convert to arrays
random.shuffle(training)
    
#create train and test lists; X - patterns, Y - intents.
#extract columns, not rows
train_x = np.array([t[0] for t in training], dtype=np.float32)
train_y = np.array([t[1] for t in training], dtype=np.float32)
print("Training data created!")
print(f"Train X shape: {train_x.shape}  |  Train Y shape: {train_y.shape}")

##-------- Build and train the model -------##
model = Sequential()
model.add(Dense(128, input_shape =(len(train_x[0]), ), activation = 'relu')) # i/p layer
model.add(Dropout(0.5))
model.add(Dense(64, activation ='relu')) # hidden layer
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax')) # o/p layer with softmat activation function to predict the probabilities for each intent.


#Compile model. 
#Stochastic gradient descent with Nesterov accelerated gradient used for training this model.  
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

#saving the model
model.save('models/chatbot_model.keras')

print("Model saved to models/chatbot_model.keras")
print("Vocab saved to models/words.pkl")
print("Classes saved to models/classes.pkl")