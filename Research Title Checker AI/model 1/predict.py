import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow startup logs for a cleaner UI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_assets():
    """Loads the trained model and preprocessing objects."""
    try:
        model = tf.keras.models.load_model('research_rnn_model.keras')
        
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        with open('label_encoder.pkl', 'rb') as handle:
            le = pickle.load(handle)
            
        return model, tokenizer, le
    except FileNotFoundError as e:
        print(f"Error: Missing required file - {e}")
        return None, None, None

def main():
    print("--- Research Review Type Predictor ---")
    print("Loading AI model, please wait...")
    
    model, tokenizer, le = load_assets()
    
    if model is None:
        return

    max_len = 20 # Must match the max_len used during training

    print("\nModel Loaded! Enter a Research Title to classify.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        # Ask for user input
        title = input("Enter Research Title: ").strip()

        if title.lower() in ['quit', 'exit', '']:
            break

        # 1. Convert text to sequence
        seq = tokenizer.texts_to_sequences([title])
        
        # 2. Pad sequence to match model input shape
        padded = pad_sequences(seq, maxlen=max_len)
        
        # 3. Predict
        prediction = model.predict(padded, verbose=0)
        
        # 4. Get the class with the highest probability
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx] * 100
        result_label = le.inverse_transform([class_idx])[0]

        print(f"Result: **{result_label}** ({confidence:.2f}% confidence)")
        print("-" * 30)

if __name__ == "__main__":
    main()





