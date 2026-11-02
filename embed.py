import os
from dotenv import load_dotenv

import cohere
import numpy as np
import pandas as pd

import faiss

# Load environment variables from .env if present
load_dotenv()

# Paste your API key here. Remember to not share publicly
api_key = os.environ.get("COHERE_API_KEY")
model = os.environ.get("COHERE_MODEL")

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.ClientV2(api_key)


def get_index(text):
    # Split into a list of sentences
    texts = text.split('.')
    # Clean up to remove empty spaces and new lines
    texts = [t.strip(' \n') for t in texts]

    # Get the embeddings
    response = co.embed(
        texts=texts,
        input_type="search_document",
        model=model
        )

    embeds = np.asarray(response.embeddings.float_)
    


    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    print(index.is_trained)
    index.add(np.float32(embeds))
    
    return index, texts, embeds.shape

def search(query, index, texts, number_of_results=3):
    # 1. Get the query's embedding
    query_embed = co.embed(
        texts=[query],
        input_type="search_query",
        model=model
        ).embeddings.float_[0]
    
    # 2. Retrieve the nearest neighbors
    distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results)
    
    # 3. Format the results
    texts_np = np.array(texts) # Convert texts list to numpy for easier indexing
    results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]], 'distance': distances[0]})
    
    return results

