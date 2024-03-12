import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import warnings
import os
from gtts import gTTS
from io import BytesIO

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset to get the intents and responses
with open(r'E:\sample_app\voicebot3.1.json') as file:  # Update the path accordingly
    data = json.load(file)

# Initialize and load the tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
all_patterns = [pattern for intent in data['intents'] for pattern in intent['patterns']]
tokenizer.fit_on_texts(all_patterns)

# Load the trained model
model = load_model(r'E:\sample_app\lstm1.1.3.h5')  # Update the path accordingly

# Streamlit app layout
st.title('Chatbot')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("You: ", placeholder="Say something...")

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=13, padding='post')
    prediction = model.predict(padded)
    return np.argmax(prediction)

def get_response(intent_index):
    tag = data['intents'][intent_index]['tag']
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)  # Go to the beginning of the IO bytes stream
    return audio_bytes

if user_input:
    predicted_intent_index = predict_intent(user_input)
    response = get_response(predicted_intent_index)
    
    # Store the conversation in session state
    st.session_state['chat_history'].append({"user": user_input, "bot": response})
    
    # Display the entire chat history
    for chat in st.session_state['chat_history']:
        st.text_area("You:", value=chat["user"], height=100, max_chars=None, key=None)
        st.text_area("Bot:", value=chat["bot"], height=100, max_chars=None, key=None)
    
    # Generate and display the audio for the response
    audio_bytes = text_to_speech(response)
    st.audio(audio_bytes, format='audio/mp3', start_time=0)
