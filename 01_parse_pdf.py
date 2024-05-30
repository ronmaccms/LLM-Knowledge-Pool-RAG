# This script needs a llama-parse key setup in the keys.py script to run.
from llama_parse import LlamaParse
import os
from config import *

# Parser parameters
parser = LlamaParse(
    api_key=LLAMAPARSE_API_KEY, 
    result_type="markdown",  # "markdown" or "text"
    num_workers=4,
    verbose=True,
    language="en",
)

for document in os.listdir(r"C:\Users\ARoncal\source\repos\GENAI-Joao\LLM-Knowledge-Pool-RAG\knowledge_pool"):
    #Iterate through the pdfs
    if document.endswith(".pdf"):
        filepath = os.path.join(r"C:\Users\ARoncal\source\repos\GENAI-Joao\LLM-Knowledge-Pool-RAG\knowledge_pool", document)

        # Parse the pdf
        pdf = parser.load_data(filepath)
        text = pdf[0].text

        # Save to a txt file
        output_filename = os.path.splitext(document)[0]
        output_path = os.path.join(r"C:\Users\ARoncal\source\repos\GENAI-Joao\LLM-Knowledge-Pool-RAG\knowledge_pool", f"{output_filename}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"Finished parsing {document}")
    
print("Finished parsing all documents")