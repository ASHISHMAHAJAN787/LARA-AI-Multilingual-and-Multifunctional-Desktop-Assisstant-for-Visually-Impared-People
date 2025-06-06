import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the tokenizer
with open('Data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('Data/label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the trained model
model = load_model('Data/chat_model.keras')

# Load the intents data
with open('Data/intents.json', 'r') as file:
    intents = json.load(file)

# Prepare data for training
texts = []
labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# Tokenize the input data
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=20)  # Adjust the maxlen if necessary

# Encode the labels
y = label_encoder.transform(labels)
y = to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

# Calculate metrics
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
accuracy = accuracy_score(y_true_classes, y_pred_classes)

print(f"INTENT RECOGNITION")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy*100:.4f}%")





# Speech Recognition Evaluation
from jiwer import wer, cer

# Example transcriptions
ground_truth_speech = "Recognize this object"
predicted_speech = "recognize this object"

# Calculate WER and CER
wer_score = wer(ground_truth_speech, predicted_speech)
cer_score = cer(ground_truth_speech, predicted_speech)
print(f"Word Error Rate (WER): {wer_score:.4f}")
print(f"Character Error Rate (CER): {cer_score:.4f}")

# Text-to-Speech Evaluation


import io
import pygame
import speech_recognition as sr
from gtts import gTTS
import difflib  # For WER calculation
import time
import tempfile
import os

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def speak(text, language_code='en'):
    """Use gTTS to speak the given text."""
    try:
        print(f"Speaking: {text}")
        tts = gTTS(text=text, lang=language_code)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        pygame.mixer.music.load(audio_fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        return audio_fp
    except Exception as e:
        print(f"Error in speaking: {e}")
        return None

def record_audio():
    """Record audio input from the user."""
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        print("Listening...")
        audio = recognizer.listen(mic)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"API request error: {e}")
            return None

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER) between reference and hypothesis texts."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    wer = 1 - matcher.ratio()  # Simplified WER
    return wer

def test_speech_recognition():
    """Test speech recognition accuracy with a known phrase."""
    reference_phrase = "the quick brown fox jumps over the lazy dog"
    speak(f"Please say: {reference_phrase}")
    time.sleep(1)
    
    recognized_text = record_audio()
    if recognized_text:
        wer = calculate_wer(reference_phrase, recognized_text)
        accuracy = (1 - wer) * 100
        print(f"Reference: {reference_phrase}")
        print(f"Recognized: {recognized_text}")
        print(f"Word Error Rate: {wer:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        speak(f"Your speech recognition accuracy was {accuracy:.2f} percent.")
    else:
        print("No text recognized, cannot calculate accuracy.")
        speak("Sorry, I couldn't recognize what you said.")

def test_text_to_speech():
    """Test text-to-speech by generating audio, recording it, and checking recognition accuracy."""
    test_phrase = "This is a test of the text to speech system."
    
    # Step 1: Generate TTS audio and save to a temporary file
    print(f"Generating TTS for: {test_phrase}")
    tts = gTTS(text=test_phrase, lang='en')
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_audio_file.name)
    temp_audio_file.close()  # Explicitly close the file to ensure it's written
    time.sleep(0.5)  # Small delay to ensure file is fully written
    
    # Step 2: Verify file exists and play it
    if os.path.exists(temp_audio_file.name):
        print(f"Temp file created: {temp_audio_file.name}")
        print("Playing TTS audio, ensure speakers are on and microphone is ready...")
        try:
            pygame.mixer.music.load(temp_audio_file.name)
            pygame.mixer.music.play()
            time.sleep(1)  # Delay to ensure playback starts
        except pygame.error as e:
            print(f"Pygame error: {e}")
            os.unlink(temp_audio_file.name)
            return
    else:
        print(f"Error: Temporary file {temp_audio_file.name} not found.")
        return
    
    # Step 3: Record the played audio
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        print("Recording TTS output...")
        audio = recognizer.listen(mic, timeout=5, phrase_time_limit=10)
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    
    # Step 4: Recognize the recorded audio
    try:
        recognized_text = recognizer.recognize_google(audio)
        print(f"Recognized from TTS: {recognized_text}")
    except sr.UnknownValueError:
        recognized_text = None
        print("Could not understand TTS audio")
    except sr.RequestError as e:
        recognized_text = None
        print(f"API request error: {e}")
    
    # Step 5: Calculate WER and accuracy
    if recognized_text:
        wer = calculate_wer(test_phrase.lower(), recognized_text.lower())
        accuracy = (1 - wer) * 100
        print(f"Original TTS Text: {test_phrase}")
        print(f"Recognized Text: {recognized_text}")
        print(f"TTS Intelligibility Accuracy: {wer*100:.2f}%")
        print(f"TTS word error: {accuracy:.2f}%")
        speak(f"The text-to-speech intelligibility accuracy was {accuracy:.2f} percent.")
    else:
        print("No text recognized from TTS output.")
        speak("Sorry, I couldn't evaluate the text-to-speech output.")
    
    # Clean up temporary file
    if os.path.exists(temp_audio_file.name):
        os.unlink(temp_audio_file.name)

def main():
    print("Starting Speech Recognition and Text-to-Speech Tests")
    speak("Hello, let's test the speech recognition and text-to-speech systems.")
    
    # Test Speech Recognition
    print("\n--- Speech Recognition Test ---")
    test_speech_recognition()
    
    # Test Text-to-Speech
    print("\n--- Text-to-Speech Test ---")
    test_text_to_speech()
    
    speak("Testing complete. Goodbye.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated.")



