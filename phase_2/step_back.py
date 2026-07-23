from google import genai
from google.genai import types
from dotenv import  load_dotenv
load_dotenv()
client = genai.Client()

def step_back_query(original_query):

    
    prompt = f"""Your task is to generate a more general, abstract version of the 
    following question. The general version should retrieve background knowledge 
    that helps answer the specific question.

    Specific question: {original_query}

    General/abstract version (one sentence):"""
    '''
    response = client.chat.completions.create(
        model = "Qwen/Qwen2.5-72B-Instruct",
        messages = [{"role": "user", "content":prompt}],
        max_tokens = 100
    )
    '''
    response = client.models.generate_content(
        model = "gemini-3.5-flash",
        contents = prompt 
    )
    
    response_text = response.text

    step_back = response_text.strip()
    print(f"original query is : {original_query}")
    print(f"general query is : {step_back}")
    
    return original_query, step_back