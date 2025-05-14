import os

from dotenv import load_dotenv
from groq import Groq

# Load Environment values
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Do you know Grok AI made by Elon Musk? If yes, do you think I could fine-tune you to reflect his sarcasm, humor and overall style?",
        }
    ],
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    #model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)

'''
def handle_groq_query(query: str) -> str:
    # Placeholder logic for Groq processing
    # Replace this with actual Groq logic
    if "true" in query.lower():
        return "Yes, it seems true."
    return "Obviously not."
'''