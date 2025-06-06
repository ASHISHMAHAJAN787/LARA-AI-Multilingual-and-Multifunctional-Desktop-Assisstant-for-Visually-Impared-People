import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load intents data (ensure the path is correct)
with open('Data/intents.json') as file:  # Adjust the path if necessary
    intents = json.load(file)

# Extract patterns and tags from the dataset
patterns = []
tags = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Tokenize the text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, maxlen=20)

# Encode the tags (labels)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Convert labels to categorical
labels = tf.keras.utils.to_categorical(labels, num_classes=len(set(tags)))

# Define the model architecture
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=20),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(len(set(tags)), activation='softmax')  # Number of output classes is the number of intents
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=500)

# Save the trained model as chat_model.h5
# Save the trained model as chat_model.keras
model.save('Data/chat_model.keras')  # Using the recommended format


# Also, save the tokenizer and label encoder for later use
import pickle

# Save tokenizer
with open('Data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save label encoder
with open('Data/label_encoder.pickle', 'wb') as enc_file:
    pickle.dump(label_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and preprocessing files have been saved successfully.")
