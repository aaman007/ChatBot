import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import random
import json
import pickle


######################################################
### Data Preprocessing / Loading Preprocessed Data
######################################################

with open("intents.json") as file:
    data = json.load(file)['intents']

try:
    with open("data.pickle", "rb") as saved_file:
	print("Loading the preprocessed data........")
	words, labels, train_x, train_y = pickle.load(saved_file)
except:
    print("Preprocessing the data.............")

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data:
        if intent['tag'] not in labels:
	    labels.append(intent['tag'])

	for pattern in intent['patterns']:
	    wrds = nltk.word_tokenize(pattern)
	    wrds = [stemmer.stem(w.lower()) for w in wrds if w != '?']
	    words.extend(wrds)
	    docs_x.append(wrds)
	    docs_y.append(intent['tag'])

    words = sorted(list(set(words)))
    labels = sorted(labels)

## Constructing training set for NN

    train_x = []
    train_y = []

    ouput_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
	bag = []
	for word in words:
	    if word in doc:
		bag.append(doc.count(word))
	    else:
                bag.append(0)

	output_row = ouput_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	train_x.append(bag)
	train_y.append(output_row)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    with open("data.pickle", "wb") as new_file:
    pickle.dump((words, labels, train_x, train_y), new_file)


###########################################
###    Training Model / Loading Model
###########################################

try:
    model = tf.keras.models.load_model('saved_models/myModel')
    print("Loading the trainded model..............")
except:
    print("Training the model..................")

    class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
	    if logs.get('acc') >= 0.98:
		self.model.stop_training = True
		
    callbacks = myCallback()
    #print(np.array(words).shape)
    model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(units=len(words), input_shape=[len(words)]),
	tf.keras.layers.Dense(12, activation="relu"),
	tf.keras.layers.Dense(20, activation="relu"),
	tf.keras.layers.Dense(len(labels), activation="softmax")
    ])

    # print(model.summary())
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
    model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=0, callbacks=[callbacks])

    # print(model.evaluate(train_x, train_y))  # Shows (loss, accuracy) as a tuple

    model.save("saved_models/myModel")


############################################
### Predicting / Chatbot in action
############################################

def bag_of_words(sentence, words):
    bag = []
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(w.lower()) for w in s_words if w != '?']

    for word in words:
	if word in s_words:
	    bag.append(s_words.count(word))
	else:
	    bag.append(0)
    return np.array([bag])

def chat():
    print("Bot is ready to talk...[ Enter quit to stop chatting]")

    while True:
	sentence = input("You : ")
	if sentence.lower() == "quit":
	    break

	bag = bag_of_words(sentence, words)
	result = model.predict(bag)[0]
	result_index = np.argmax(result)
	tag = labels[result_index]
	#print(result)
	#print(result_index)
	if result[result_index] > 0.7:
	    for intent in data:
		if intent['tag'] == tag:
		    responses = intent['responses']

	    print("Bot : ", random.choice(responses))
	else:
	    print("Sorry. I don't quite understand what you are saying. Try asking a different question.")

chat()
