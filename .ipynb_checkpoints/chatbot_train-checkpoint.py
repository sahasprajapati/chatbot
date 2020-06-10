import nltk
from nltk.stem import WordNetLemmatizer

import json
import pickle
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

if __name__ == '__main__':
    # Data Loading
    lemmatizer = WordNetLemmatizer()
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']
    data_file = open('intents.json').read()
    intents = json.loads(data_file)
    
    # Data Preprocessing
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            
            documents.append((w, intent['tag']))
            
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    
    classes = sorted(list(set(classes)))
    
    print(f'{len(words)} words')
    print(f'{len(classes)} classes')
    print(f'{len(documents)} documents')
    
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    
    
    # Dataset generation
    training = []
    output_empty = [0] * len(classes)
    #print(output_empty)
    for doc in documents:
        # bag of words (BOW)
        bag = []
        pattern_words = doc[0]

        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

        #print(training)
        #print(output_row)
        #print(bag)
        #print(pattern_words)

    random.shuffle(training)
    training = np.array(training)
    # X-Pattern
    train_x = list(training[:,0])
    # Y-Intents
    train_y = list(training[:,1])
    print('Training data created')
    
    
    # Model generation
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)

    print('model generated')