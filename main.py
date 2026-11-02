import embed as emb
import bm25search
from tqdm import tqdm

text = """
    Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
    It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
    Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.
    Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
    Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
    Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
    Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
    Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.
    Interstellar premiered on October 26, 2014, in Los Angeles.
    In the United States, it was first released on film stock, expanding to venues using digital projectors.
    The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
    It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
    It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
    Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades
"""

index, texts, shape = emb.get_index(text)

print(f"Number of sentences indexed: {len(texts)}")
print(f"Index created with shape: {shape}")

# Check Model Embedding Size: https://docs.cohere.com/docs/cohere-embed

query = "how precise was the science"
# results = emb.search(query, index, texts)

# print(f"Query:'{query}'\nNearest neighbors:")
# print(results)

# Comparing BM25 and Embedding Search Results
tokenized_corpus = []

for passage in tqdm(texts):
    tokenized_corpus.append(bm25search.bm25_tokenizer(passage))

# bm25 = bm25search.BM25Okapi(tokenized_corpus)
# bm25search.keyword_search(query = "how precise was the science", bm25=bm25, texts=texts)

