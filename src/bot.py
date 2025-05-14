from twitchio.ext import commands
import os
from dotenv import load_dotenv
from src.groq_handler import handle_groq_query

# Load Environment values
load_dotenv()

# Initialize the Bot
class Bot(commands.Bot):
    
    def load_system_prompt(self, filename: str) -> str:
        try:
            with open(filename, "rt", encoding = 'utf-8', errors='ignore') as file:
                extract = file.read()
                return extract
        
        except FileNotFoundError:
            print(f"{filename} does not exist, working with an empty one.")
            with open(filename, "wt", encoding = 'utf-8', errors='ignore') as file:
                file.write("You are Grok, a witty and sarcastic AI assistant.")
            
            with open(filename, "rt", encoding = 'utf-8', errors='ignore') as file:
                return file.read()
        
        except Exception as e:
            print(f"Loading context was unsuccessful\n{e}")
                
    
    def __init__(self):
        super().__init__(
            token=os.getenv("AccessToken"),
            prefix='!',
            initial_channels=["yeetzgaming20"]
        )
        
        # Read system prompt from file
        self.system_prompt = self.load_system_prompt("sysprompt.txt")
        
        # Error checking
        if not self.system_prompt:
            print("Warning: system prompt is empty. Using a default prompt.")
            self.system_prompt = "You are Grok, a witty and sarcastic AI assistant."

        

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print('Bot is ready!')

    async def event_message(self, message):
        if message.echo:
            return
        
        if "@grok is this true?" in message.content.lower():
            response = handle_groq_query(system_prompt = self.system_prompt, query = message.content)
            await message.channel.send(f"Hello @{message.author.name}. {response}")