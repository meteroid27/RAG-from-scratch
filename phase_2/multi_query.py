from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'Qwen/Qwen2.5-72B-Instruct',
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm )


def generate_multi_query(original_query, n):
    
    prompt  = f"""Generate {n} different search queries for the same question.
    Each should use different keywords and phrasing.
    Return ONLY the queries, one per line, no numbering or extra text.
    Original question: {original_query}
    {n} alternative queries:"""

    response = model.invoke(prompt).content
    
    varients = response.strip().split("\n")
    
    result = [v.strip() for v in varients if v.strip()]

    all_query = original_query + result
    print(all_query)
    return all_query
    
original_query = ["what are they talking about dopamine?"]

generate_multi_query(original_query, 3)

    

    
    
    
    


    