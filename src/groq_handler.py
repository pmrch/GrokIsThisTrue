import os

from dotenv import load_dotenv
from groq import Groq

# Load Environment values
load_dotenv()

# Initialize the Groq client only once
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

# Predefine character limti
MAX_TOKENS = 125

def handle_groq_query(system_prompt: str, query: str, user_name: str) -> str:
    cleaned_query = (
    f"{query.replace('@grok is this true?', '').strip()}\n\n"
    #f"Context from Neuro:\n{Neuro.strip()}\n\n"
    "My friend said AI are taking over."
    f"Grok, is this true?"
    )

    
    try:
        # Make the API call to Groq's chat completion
        chat_completion = client.chat.completions.create(
            messages = [{
                "role": "system",
                "content": system_prompt,
            },{
                "role": "user",
                "content": cleaned_query,
                "name": user_name,
            }
        ], 
        model="llama-3.3-70b-versatile",
        max_completion_tokens = MAX_TOKENS)
        
        return f"{chat_completion.choices[0].message.content}"
    
    except Exception as e:
        return f"This shit doesn't work.\n{e}"
    
