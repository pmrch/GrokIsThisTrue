# -*- coding: utf-8 -*-
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

# Configuration for the API request
CONFIG = {
    "MAX_TOKENS": 125,
    "MODEL": "llama-3.3-70b-versatile",
    "PROMPT_PATTERN": re.compile(r"@grok(?:ai1)?[, ]*is (?:this|that) true\??", re.IGNORECASE),
}

def handle_groq_query(system_prompt: str, query: str, user_name: str, allow_unmatched: bool = True) -> str:
    """
    Sends a cleaned and reformatted query to the Groq API based on Twitch chat input.
    
    Args:
        system_prompt (str): The base system instruction for the AI.
        query (str): Raw chat message from the user.
        user_name (str): Twitch username of the message sender.
        
    Returns:
        str: AI response or error string.
    """
    
    cleaned_query = str()
    
    match = CONFIG["PROMPT_PATTERN"].search(query)
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
        model=CONFIG["MODEL"],
        max_completion_tokens=CONFIG["MAX_TOKENS"])
        
        return f"{chat_completion.choices[0].message.content}"
    
    except (IndexError, AttributeError) as e:
        print(f"[ERROR] Response parsing issue: {e}")
        return "Couldn't understand Groq's reply this time, sorry."
    
    except Exception as e:
        return f"Unexpected error encountered.\n{e}"
    
