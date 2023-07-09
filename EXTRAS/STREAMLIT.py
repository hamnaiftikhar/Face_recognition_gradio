# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:48:10 2023

@author: hamna
"""

import gradio as gr
import time
import streamlit as st


def replace(text):
    return text.replace('World', 'Databricks')

st.title("Text Replacement")
    input_text = st.text_area("Enter text", value="Hello, World!")
    output_text = replace(input_text)
    st.text_area("Replaced text", value=output_text)
