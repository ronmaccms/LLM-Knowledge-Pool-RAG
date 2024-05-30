# This script runs locally (w/LM Studio server) and an embedding model loaded.
import json
import os
from config import *

documents_directory = r"C:\Users\ARoncal\source\repos\GENAI-Joao\LLM-Knowledge-Pool-RAG\knowledge_pool"

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return local_client.embeddings.create(input = [text], model=model).data[0].embedding

# Iterate over all .txt files in the directory
for filename in os.listdir(documents_directory):
    if filename.endswith('.txt'):
        document_to_embed = os.path.join(documents_directory, filename)

        # Read the text document
        with open(document_to_embed, 'r', encoding='utf-8', errors='ignore') as infile:
            text_file = infile.read()

        # Split the text into lines (each line = 1 vector). Pick this or the following chunking strategy.
        chunks = text_file.split("\n")
        chunks = [line for line in chunks if line.strip() and line.strip() != '---']

        # Create the embeddings
        embeddings = []
        for i, line in enumerate(chunks):
            print(f'{i} / {len(chunks)}')
            vector = get_embedding(line.encode(encoding='utf-8').decode())
            database = {'content': line, 'vector': vector}

        # Save the embeddings to a json file
        output_filename = os.path.splitext(document_to_embed)[0]
        output_path = f"{output_filename}.json"

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(embeddings, outfile, indent=2, ensure_ascii=False)

        print(f"Finished vectorizing. Created {document_to_embed}")