import os
import json
from config import *

# Specify the directory you want to embed
directory_to_embed = "C:/Users/ARoncal/source/repos/GENAI-Joao/knowledge_pool"  # Replace with your actual directory path

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return local_client.embeddings.create(input = [text], model=model).data[0].embedding

# Loop over the files in the directory
for filename in os.listdir(directory_to_embed):
    # Only process .txt files
    if filename.endswith(".txt"):
        document_to_embed = os.path.join(directory_to_embed, filename)

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
            embeddings.append(database)

        # Save the embeddings to a json file
        output_filename = os.path.splitext(document_to_embed)[0]
        output_path = f"{output_filename}.json"

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(embeddings, outfile, indent=2, ensure_ascii=False)

        print(f"Finished vectorizing. Created {output_path}")