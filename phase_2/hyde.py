from google import genai
from google.genai import types
import os

client = genai.Client()

def generate_hyde(original_query , embedding_model):
    hyde_prompt = f"""Write a short paragraph (3-4 sentences) that directly answers 
    this question, as if you were an expert. This is for search purposes, not the final answer.

    Question: {original_query}

    Answer:"""
    
    response = client.models.generate_content(
        model = "gemini-3.5-flash",
        content = hyde_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=150
        )
    )
    
    hypo_answer = response.text
    
    embeddings = embedding_model([hypo_answer])[0]
    
    
    return embeddings , hypo_answer
