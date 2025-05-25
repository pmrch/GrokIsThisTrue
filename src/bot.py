import os
import aiofiles # type: ignore
import regex as re

from datetime import datetime, timedelta
from twitchio.ext import commands
from dotenv import load_dotenv

from src.transcriber import Transcriber
from src.groq_handler import handle_groq_query
from src.helper_functions import *

# Load Environment values
load_dotenv()

# Initialize the Bot
class Bot(commands.Bot):
    def load_system_prompt(self, filename: str):
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
        
    async def select_context(self):
        os.makedirs("data/logs", exist_ok=True)

        file1_lines = await load_file_lines_async(str("data/transcript.txt"))
        file2_lines = await load_file_lines_async(str("data/logs/chat_log.txt"))

        base_ts, _, base_line = find_line_by_pattern(file2_lines, self.pattern)

        if base_ts:
            timestamp_format = "%Y-%m-%d %H:%M:%S"

            base_ts_dt = datetime.strptime(base_ts, timestamp_format)
            target_ts = base_ts_dt - timedelta(seconds=30)
            
            _, _, closest_line = find_closest_line(target_ts, file1_lines)

            # Debug logging
            async with aiofiles.open("data/debug.txt", "a", encoding="utf-8") as debug:
                await debug.write(f"Base line: {base_line}\n")
                await debug.write(f"Closest line: {closest_line}\n\n")

            return base_line, closest_line
        else:
            async with aiofiles.open("data/debug.txt", "a", encoding="utf-8") as debug:
                await debug.write("No line matched pattern in file2\n")

            return None, None
        
    def __init__(self):
        super().__init__(
            token = os.getenv("AccessToken", ""),
            prefix = '!',
            initial_channels = ["yeetzgaming20", "vedal987"]
        )
        
        # Read system prompt from file
        self.system_prompt = self.load_system_prompt("data/sysprompt.txt")
        
        # Define the filename of chatlog 
        self.chatlog_file = "data/logs/chat_log.txt"
        
        # Make message detection dynamic
        self.pattern = re.compile(r"@grok(?:ai1)?[, ]*is (?:this|that) true\??", re.IGNORECASE)
        
        # Error checking
        if not self.system_prompt:
            print("Warning: system prompt is empty. Using a default prompt.")
            self.system_prompt = "You are Grok, a witty and sarcastic AI assistant."

    async def event_ready(self) -> None:
        print(f'Logged in as | {self.nick}')
        print('Bot is ready!')
        
        # Start transcription process in the background threads
        self.transcriber = Transcriber()
        self.transcriber.start()

    async def event_message(self, message):
        if message.echo:
            return
        
        matching = re.search(pattern = self.pattern, string = str(message.content).lower())
        if matching:
            base_line, closest_line = await self.select_context()
            
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