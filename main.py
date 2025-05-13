import os
import time
import twitchio
import asyncio

from twitchio.ext import commands
from dotenv import load_dotenv

# Load Environment values
load_dotenv()

# Create basic bot
class Bot(commands.Bot):
    
    def __init__(self):
        # Initialise Bot with details
        super().__init__(
            token=f"{os.getenv("AccessToken")}",
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
            await message.channel.send(f"Hello @{message.author.name}. Obviously not")
        
        
bot = Bot()
bot.run()