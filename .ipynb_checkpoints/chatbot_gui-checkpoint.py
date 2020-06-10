import nltk 
from nltk.stem import WordNetLemmatizer

import pickle
import numpy as np
import json
import random

import tkinter
from tkinter import Tk, Button, Text, Scrollbar
from tensorflow.keras.models import load_model

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Foud in bag: {w}")
    return(np.array(bag))
                    
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res
    
    
    
def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)
    
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, 'You: ' + msg + '\n\n')
        
        res = chatbot_response(msg)
        ChatLog.insert(END, 'Bot: ' + res + '\n\n')
        
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    
    base = Tk()
    base.title('Hello')
    base.geometry('400x500')
    base.resizable(width=FALSE, height=FALSE)

    ChatLog = Text(base)
    ChatLog.config(state=DISABLED)

    scrollbar = Scrollbar(base)
    ChatLog['yscrollcommand'] = scrollbar.set

    SendButton = Button(base, text='Send', command=send)

    EntryBox = Text(base)

    scrollbar.place(x=376, y=6, height=386)
    ChatLog.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)

    base.mainloop()
    