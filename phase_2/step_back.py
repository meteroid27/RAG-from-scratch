from huggingface_hub import InferenceClient
from dotenv import load_dotenv
client = InferenceClient()


load_dotenv()

def step_back_query(original_query):

    
    prompt = f"""Your task is to generate a more general, abstract version of the 
    following question. The general version should retrieve background knowledge 
    that helps answer the specific question.

    Specific question: {original_query}

    General/abstract version (one sentence):"""
    
    response = client.chat.completions.create(
        model = "Qwen/Qwen2.5-72B-Instruct",
        messages = [{"role": "user", "content":prompt}],
        max_tokens = 100
    )
    
    step_back = response.choices[0].message.content.strip()
    print(f"original query is : {original_query}")
    print(f"general query is : {step_back}")
    
    return original_query, step_back


query = "what are they talking about the dopamine in the video?"

step_back_query(query)