"""
Helper script to create tokenizer.pkl from your training data
Run this if you don't have a tokenizer.pkl file
"""

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer_from_text(text_file_path):
    """
    Create and save a tokenizer from a text file
    
    Args:
        text_file_path: Path to your novel/text file used for training
    """
    print(f"Reading text from {text_file_path}...")
    
    # Read the text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Tokenizer saved as tokenizer.pkl")
    print("\nSample words from vocabulary:")
    for i, (word, idx) in enumerate(list(tokenizer.word_index.items())[:10]):
        print(f"  {word}: {idx}")

if __name__ == "__main__":
    # Replace 'your_novel.txt' with your actual text file
    text_file = input("Enter the path to your text file (e.g., novel.txt): ")
    
    try:
        create_tokenizer_from_text(text_file)
    except FileNotFoundError:
        print(f"Error: File '{text_file}' not found!")
    except Exception as e:
        print(f"Error: {e}")
