import os

from dotenv import load_dotenv
from groq import Groq

# Load Environment values
load_dotenv()

# Initialize the Groq client only once
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def handle_groq_query(system_prompt: str, query: str) -> str:
    try:
        # Make the API call to Groq's chat completion
        chat_completion = client.chat.completions.create(
            messages = [{
                "role": "system",
                "content": system_prompt,
            },{
                "role": "user",
                "content": query,
            }
        ], 
        model="llama-3.3-70b-versatile",)
        
        return f"{chat_completion.choices[0].message.content}"
    
    except Exception as e:
        return f"This shit doesn't work.\n{e}"
    
