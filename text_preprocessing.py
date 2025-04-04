import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

import numpy as np
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens






import numpy as np

# Function to get the average word embeddings for a list of tokens


def nlkt_text_preprocessing(train_df):
    # Download NLTK stopwords if not already downloaded
    nltk.download('stopwords')
    nltk.download('punkt')

    # Load the recipe data
    recipe_path = 'recipe.tsv'
    recipes_df = pd.read_csv(recipe_path, sep='\t')

    # Merge the train_df with recipes_df to get the relevant recipes
    train_recipes = train_df[['item_id']].merge(recipes_df[['item_id', 'name', 'description']], on='item_id', how='left')

    # Concatenate the name and description fields
    train_recipes['text'] = train_recipes['name'].fillna('') + ' ' + train_recipes['description'].fillna('')

    # Define a function for text preprocessing


    # Apply text preprocessing to the text field
    train_recipes['tokens'] = train_recipes['text'].apply(preprocess_text)

    # Flatten the list of tokens to calculate the vocabulary size
    all_tokens = [token for tokens in train_recipes['tokens'] for token in tokens]
    vocab_size = len(set(all_tokens))

    print(f'Vocabulary Size after preprocessing: {vocab_size}')



