import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the CSV
try:
    df_research = pd.read_csv('research_reviews2.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'research_reviews2.csv' not found. Please create the file first.")
    exit()

# 2. Preprocessing
max_words = 1000
max_len = 20

tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df_research['Research Title'])

X = tokenizer.texts_to_sequences(df_research['Research Title'])
X = pad_sequences(X, maxlen=max_len)

le = LabelEncoder()
y_encoded = le.fit_transform(df_research['Review Type'])
y = tf.keras.utils.to_categorical(y_encoded)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the RNN (LSTM) Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 64, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(64, dropout=0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train
print("Starting training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model on the 20% of data it has never seen (test set)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

# # 5. Predict a New Title
# import pickle

# # 1. Save the RNN Model (TensorFlow native format)
# model.save('research_rnn_model.keras')

# # 2. Save the Tokenizer (Pickle format)
# with open('tokenizer.pkl', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # 3. Save the Label Encoder (Pickle format)
# with open('label_encoder.pkl', 'wb') as handle:
#     pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("Model and preprocessing objects saved successfully!")