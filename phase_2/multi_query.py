import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from google import genai
from google.genai import types

from dotenv import load_dotenv 
load_dotenv()

client = genai.Client()

def generate_multi_query(original_query, n):
    
    prompt  = f"""Generate {n} different search queries for the same question.
    Each should use different keywords and phrasing.
    Return ONLY the queries, one per line, no numbering or extra text.
    Original question: {original_query}
    {n} alternative queries:"""
    
    response = client.models.generate_content(
        model = "gemini-3.6-flash",
        contents = prompt
        )   
    
    varients = response.text.strip().split("\n")
         
    result = [v.strip() for v in varients if v.strip()]
 
    all_query = [original_query] + result
    print(all_query)
    return all_query


        

    

    

    

    





    


