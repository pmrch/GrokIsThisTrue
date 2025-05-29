# -*- coding: utf-8 -*-
import os
import aiofiles # type: ignore
import regex as re

from datetime import datetime
from twitchio.ext import commands
from dotenv import load_dotenv

from src.groq_handler import handle_groq_query
from src.helper_functions import *

# Load Environment values
load_dotenv()

# Initialize the Bot
class Bot(commands.Bot):    
    def __init__(self):
        super().__init__(
            token = os.getenv("AccessToken", ""),
            prefix = '!',
            initial_channels = ["yeetzgaming20", "vedal987"]
        )
        
        # Read system prompt from file
        self.system_prompt = load_system_prompt("data/sysprompt.txt")
        
        # Define the filename of chatlog 
        self.chatlog_file = "data/logs/chat_log.txt"
        
        # Make message detection dynamic
        self.pattern = re.compile(r"@grok(?:ai1)?[, ]*is (?:this|that) true\??", re.IGNORECASE)
        
        # Declare select_context()
        self.select_context = select_context
        
        # Error checking
        if not self.system_prompt:
            print("Warning: system prompt is empty. Using a default prompt.")
            self.system_prompt = "You are Grok, a witty and sarcastic AI assistant."

    async def event_ready(self) -> None:
        print(f'Logged in as | {self.nick}')
        print('Bot is ready!')
        
        # Start transcription process in the background threads
        '''
        Yet to be implemented:
        - Create a copy of Transcriber class as object
        - Start the transcription process in a background thread
        '''

    async def event_message(self, message):
        if message.echo:
            return
        
        matching = re.search(pattern = self.pattern, string = str(message.content).lower())
        if matching:
            base_line, closest_line = await self.select_context(self.pattern)
            
            context_str = f"{base_line or ''}\n{closest_line or ''}"
            response = handle_groq_query(
                system_prompt = str(self.system_prompt), 
                query = f"Context: {str(context_str)}\nQuestion: {str(message.content)}\n", 
                user_name = str(message.author.name)
            )
            await message.channel.send(f"Hello @{message.author.name}. {response}")
        
        async with aiofiles.open(self.chatlog_file, "a", encoding = "utf-8", errors = "ignore") as log:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            await log.write(f"[{timestamp}] {message.author.name}: {message.content}\n")