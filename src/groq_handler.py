import os, re

from dotenv import load_dotenv
from groq import Groq

# Load Environment values
load_dotenv()

# Initialize the Groq client only once
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Missing GROQ_API_KEY environment variable.")
client = Groq(api_key = api_key)

PROMPT_PATTERN = re.compile(r"@grok(?:ai1)?[, ]*is (?:this|that) true\??", re.IGNORECASE)

# Predefine character limti
MAX_TOKENS = 125

def handle_groq_query(system_prompt: str, query: str, user_name: str) -> str:
    cleaned_query = str()
    
    match = PROMPT_PATTERN.search(query)
    if not match:
        print("Pattern not matched in query. Sending raw content.")
        cleaned_query = query
    else:
        cleaned_query = query.replace(match.group(), "").strip()
        cleaned_query = f"Neuro said {cleaned_query}.\nIs this true?"
    
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
        max_completion_tokens=MAX_TOKENS)
        
        return f"{chat_completion.choices[0].message.content}"
    
    except (IndexError, AttributeError) as e:
        return f"Groq response parsing failed: {e}"
    
    except Exception as e:
        return f"Unexpected error encountered.\n{e}"
    
