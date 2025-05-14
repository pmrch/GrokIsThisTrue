from twitchio.ext import commands
import os
from dotenv import load_dotenv
from src.groq_handler import handle_groq_query

# Load Environment values
load_dotenv()

# Initialize the Bot
class Bot(commands.Bot):
    
    def __init__(self):
        super().__init__(
            token=os.getenv("AccessToken"),
            prefix='!',
            initial_channels=["yeetzgaming20"]
        )

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print('Bot is ready!')

    async def event_message(self, message):
        if message.echo:
            return
        
        if "@grok is this true?" in message.content.lower():
            response = self.groq_handler.process_query(message.content)
            await message.channel.send(f"Hello @{message.author.name}. {response}")