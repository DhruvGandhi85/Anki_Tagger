import os
import openai
import pandas as pd
import tiktoken
from tqdm import tqdm
import time

# OpenAI Configuration
OPENAI_API_KEY_ENV_VAR = 'OPENAI_API_KEY'
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"
MAX_TOKENS = 8000
BATCH_SIZE = 50  # Adjust based on your needs and API rate limits

def set_api_key(api_key):
    openai.api_key = api_key

def load_dataset(input_datapath):
    assert os.path.exists(input_datapath), f"{input_datapath} does not exist. Please check your file path."
    df = pd.read_csv(input_datapath, sep='\t', header=None, usecols=[0,1], names=["guid", "card"], comment='#').dropna()
    return df

def filter_by_tokens(df, encoding):
    df["tokens"] = df.card.apply(lambda x: len(encoding.encode(x)))
    return df[df.tokens <= MAX_TOKENS]

def calculate_embeddings(df):
    embeddings = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Calculating embeddings", dynamic_ncols=True):
        batch = df.iloc[i:i+BATCH_SIZE]
        batch_embeddings = []
        for card in batch.card:
            try:
                response = openai.Embedding.create(input=card, model=EMBEDDING_MODEL)
                emb = response['data'][0]['embedding']
                batch_embeddings.append(emb)
            except Exception as e:
                print(f"Error getting embedding for card: {e}")
                batch_embeddings.append(None)
        embeddings.extend(batch_embeddings)
        time.sleep(1)  # Avoid hitting rate limits
    return embeddings

def save_embeddings(df, embeddings, output_prefix):
    df["emb"] = embeddings
    df.to_csv(f"./{output_prefix}_embeddings.csv", index=False)

def main():
    api_key = os.environ.get(OPENAI_API_KEY_ENV_VAR)
    assert api_key, f"Set your OpenAI API key as an environment variable named '{OPENAI_API_KEY_ENV_VAR}'"

    # Set OpenAI API key
    set_api_key(api_key)

    # Set deck to embed.
    input_datapath = "./anki.txt"
    output_prefix = "anki" # EDIT AS NEEDED

    # Load and preprocess dataset
    df = load_dataset(input_datapath)
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    df = filter_by_tokens(df, encoding)

    # Calculate embeddings for cards
    embeddings = calculate_embeddings(df)

    # Save embeddings to file
    save_embeddings(df, embeddings, output_prefix)

if __name__ == "__main__":
    main()
