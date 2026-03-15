from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import ttk
#importing require python packages
import os
import numpy as np
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import string
import pandas as pd
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding #loading LSTM and CNN classes
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import torch
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from transformers import pipeline
import matplotlib.pyplot as plt

main = Tk()
main.title("Utilizing Deep Learning Approaches for Text Summarization")
main.geometry("1300x1200")

global score, seq2seq_model, transformer, dataset, max_article_len, max_summary_len, summary_vocab, article_vocab
global X_train, X_test, y_train, y_test, articles, summary, max_article_len, max_summary_len, transformer

#function to clean dataset
def clean_sentence(sentence):
    lower_case_sent = sentence.lower()
    string_punctuation = string.punctuation + "¡" + '¿'
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
    return clean_sentence

#function to tokenize phrases from sentnces
def tokenize(sentences):
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

def uploadDataset():
    global dataset, records, filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,nrows=50)
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    global dataset, articles, summary, max_article_len, max_summary_len, summary_vocab, article_vocab
    text.delete('1.0', END)
    #now extract articles and summary from the dataset and then find max length of articles and summary
    articles = dataset['article'].ravel()
    summary = dataset['highlights'].ravel()
    article_text_tokenized, article_text_tokenizer = tokenize(articles)
    summary_text_tokenized, summary_text_tokenizer = tokenize(summary)
    text.insert(END,'Maximum Article Length: {}'.format(len(max(article_text_tokenized,key=len)))+"\n")
    text.insert(END,'Maximum Summary Length: {}'.format(len(max(summary_text_tokenized,key=len)))+"\n\n")

    #now extract vocabulary or unique words from the sentences
    article_vocab = len(article_text_tokenizer.word_index) + 1
    summary_vocab = len(summary_text_tokenizer.word_index) + 1
    text.insert(END,"Article vocabulary is of {} unique words".format(article_vocab)+"\n")
    text.insert(END,"Summary vocabulary is of {} unique words".format(summary_vocab)+"\n\n")

    max_article_len = int(len(max(article_text_tokenized,key=len)))
    max_summary_len = int(len(max(summary_text_tokenized,key=len)))
    article_pad_sentence = pad_sequences(article_text_tokenized, max_article_len, padding = "post")
    summary_pad_sentence = pad_sequences(summary_text_tokenized, max_summary_len, padding = "post")
    summary_pad_sentence = summary_pad_sentence.reshape(*summary_pad_sentence.shape, 1)
    #splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(article_pad_sentence, summary_pad_sentence, test_size=0.2)
    text.insert(END,"Max Article Length : "+str(max_article_len)+"\n")
    text.insert(END,"Max Summary Length : "+str(max_summary_len)+"\n\n")

def trainSeq2Seq():
    global X_train, X_test, y_train, y_test, score, seq2seq_model, max_article_len, max_summary_len, article_vocab, summary_vocab
    global articles, summary, max_article_len, max_summary_len
    score = []
    text.delete('1.0', END)
    #now train LSTM with sequences as seq2seq model
    input_sequence = Input(shape=(max_article_len,))
    embedding = Embedding(input_dim=article_vocab, output_dim=128,)(input_sequence)
    seq2seq_model = LSTM(32, return_sequences=False)(embedding) #defining lstm input model
    r_vec = RepeatVector(max_summary_len)(seq2seq_model)
    seq2seq_model = LSTM(32, return_sequences=True, dropout=0.2)(r_vec)#defining lstm output model
    logits = TimeDistributed(Dense(summary_vocab))(seq2seq_model)
    seq2seq_model = Model(input_sequence, Activation('softmax')(logits)) #now create CNN model by using lstm input and output layer
    seq2seq_model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(1e-3), metrics=['accuracy'])
    #now train and load CNN model
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = seq2seq_model.fit(X_train, y_train, batch_size=8, epochs=5000, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)#train encoder and decoder model
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        seq2seq_model.load_weights("model/model_weights.hdf5")
    print(seq2seq_model.summary())    
    seq = T5ForConditionalGeneration.from_pretrained('t5-small')
    seq2seq_model = T5Tokenizer.from_pretrained('t5-small',model_max_length=512)
    device = torch.device('cpu')
    tokenizedText = seq2seq_model.encode(articles[0], return_tensors='pt', max_length=512, truncation=True).to(device)
    #now predict summary using CNN MODEL on given phrases
    summaryIds = seq.generate(tokenizedText, min_length=30, max_length=120)
    #now extract summary from predicted array
    predict = seq2seq_model.decode(summaryIds[0], skip_special_tokens=True)
    #calculate rouge scores on test dtaa
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary[0], predict)
    rouge1 = np.amax(scores.get('rouge1'))
    rouge2 = np.amax(scores.get('rougeL'))
    text.insert(END,"Seq2Seq Rouge Score : "+str(1-rouge1)+"\n")
    score.append(1-rouge1)

def trainTransformer():
    global articles, summary, transformer
    transformer = pipeline("summarization")
    predict = transformer(articles[0])[0]['summary_text']
    #calculate rouge scores on test dtaa
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary[0], predict)
    rouge1 = np.amax(scores.get('rouge1'))
    rouge2 = np.amax(scores.get('rougeL'))
    text.insert(END,"Transformer Rouge Score : "+str(1-rouge1)+"\n")
    score.append(1-rouge1)

def graph():
    global score
    labels = ['Seq2Seq', 'Transformer']
    height = score
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Algorithm Names")
    plt.ylabel("Rouge Score")
    plt.title("Rouge Score Comparison Between Seq2Seq & Transformers")
    plt.xticks()
    plt.tight_layout()
    plt.show()

def predictSummary():
    global transformer
    text.delete('1.0', END)
    input_text = tf1.get()
    tf1.delete(0, END)
    predict = transformer(input_text)[0]['summary_text']
    text.insert(END,"Input Test = "+input_text+"\n\n")
    text.insert(END,"Predicted Summary = "+predict)
    

font = ('times', 15, 'bold')
title = Label(main, text='Utilizing Deep Learning Approaches for Text Summarization')
title.config(bg='bisque', fg='purple1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload CNN-Mail Summary Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=350,y=100)
processButton.config(font=font1)

seqButton = Button(main, text="Train Seq2Seq Model", command=trainSeq2Seq)
seqButton.place(x=650,y=100)
seqButton.config(font=font1)

transformerButton = Button(main, text="Train Transformer Model", command=trainTransformer)
transformerButton.place(x=50,y=150)
transformerButton.config(font=font1)

graphButton = Button(main, text="Rouge Score Comparison Graph", command=graph)
graphButton.place(x=350,y=150)
graphButton.config(font=font1)

summaryButton = Button(main, text="Predict Summary from Test Paragraph", command=predictSummary)
summaryButton.place(x=650,y=150)
summaryButton.config(font=font1)

l1 = Label(main, text='Input Text:')
l1.config(font=font)
l1.place(x=50,y=200)

tf1 = Entry(main,width=70)
tf1.config(font=font)
tf1.place(x=160,y=200)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='cornflower blue')
main.mainloop()
