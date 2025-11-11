# Chatbot Project (Python + NLTK + Keras)

A simple rule-based + machine learning chatbot built using Python, NLTK, and Keras, with a Tkinter GUI for interactive conversation. This project demonstrates how natural language processing (NLP) and deep learning can be combined to create an intelligent conversational assistant.


## Features

-  GUI-based chatbot using **Tkinter**
-  Intent classification using **Keras Sequential Model**
-  Text preprocessing with **NLTK** (tokenization, lemmatization)
-  Model trained on JSON-based intents dataset
-  Model persistence using **pickle (.pkl)** and **.h5** files
-  Modular structure for easy maintenance and extension



## Project Layout:

chatbot_project/
│
├── env/                      # virtual environment (you create this locally)
├── models/                   # saved model and processed data (auto-created)
│   ├── chatbot_model.keras   # saved Keras model (created after training)
│   ├── words.pkl             # vocabulary (auto-saved by training script)
│   └── classes.pkl           # intent labels (auto-saved by training script)
│
├── data/
│   └── intents.json          # patterns & responses for the chatbot
│
├── train_chatbot.py          # script to train and save the model
├── chatbot.py                # core chatbot logic (load model, predict, respond)
└── chatbot_gui.py            # Tkinter GUI for chatting with the bot
├── requirements.txt          # Required dependencies
└── README.md                 # Project documentation

##  Setup

1. Clone/download the project.
2. Activate your virtual environment:

**Windows:**
venv\Scripts\activate

**Mac/Linux:**
source venv/bin/activate

## Install dependencies:

pip install -r requirements.txt


## Train the Model
python train_chatbot.py

This will preprocess the dataset, train the neural network, and save model.h5, words.pkl, and classes.pkl.

## Run the chatbot 
python chatbot.py
This is the core logic used by chatbot_gui. This module loads the trained assets, converts user text to BoW. Predicts intent with a confidence threshold, and returns a response from intents.json.

## Run the Chatbot GUI
python chatbot_gui.py

## How It Works
Data Source – The intents.json file contains user intents, patterns, and pre-defined responses.

Preprocessing – The text is tokenized and lemmatized using NLTK.

Vectorization – Each message is transformed into a Bag-of-Words (BoW) vector.

Model Training – A Keras Sequential model learns to classify user intents.

Inference – During conversation, the model predicts the most probable intent, and an appropriate response is displayed in the GUI.

## Example Intents
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Good morning", "Hey there"],
      "responses": ["Hello!", "Hi! How can I assist you today?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["Goodbye!", "Take care!"]
    }
  ]
}


## Technologies Used
| Category      | Tools / Libraries  |
| ------------- | ------------------ |
| Programming   | Python 3.8+        |
| NLP           | NLTK               |
| Deep Learning | Keras / TensorFlow |
| GUI           | Tkinter            |
| Serialization | Pickle             |
| Data          | JSON               |


## Future Improvements
 Add context-awareness for multi-turn conversations

 Convert to Flask / Streamlit web app for deployment

 Integrate speech recognition and TTS

 Replace bag-of-words with transformer embeddings (BERT/GPT)

 Expand intents and improve accuracy with real-world data
