# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:48:10 2023

@author: hamna
"""

import gradio as gr
import random
import nltk
from nltk.corpus import gutenberg

# Get the text from the Gutenberg dataset
dataset = gutenberg.raw()

# Build Markov chain model
def build_markov_model(dataset):
    model = {}
    sentences = nltk.sent_tokenize(dataset)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for i in range(len(words)-1):
            if words[i] not in model:
                model[words[i]] = []
            model[words[i]].append(words[i+1])
    return model

# Generate next words using Markov chain model
def generate_next_words(prompt):
    words = nltk.word_tokenize(prompt)
    if words[-1] in markov_model:
        next_word = random.choice(markov_model[words[-1]])
    else:
        next_word = "<End of Text>"
    return ' '.join(words + [next_word])

# Build Markov chain model
markov_model = build_markov_model(dataset)

iface = gr.Interface(
    fn=generate_next_words,
    inputs=gr.inputs.Textbox(placeholder="Enter text"),
    outputs=gr.outputs.Textbox(label="Generated Words")
)

iface.launch()
