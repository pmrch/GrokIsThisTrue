
# -*- coding: utf-8 -*-import re, bisect, aiofiles, os # type: ignore
from datetime import datetime, timedelta
import aiofiles, re, os, bisect


def load_system_prompt(filename: str):
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

async def select_context(pattern):
        os.makedirs("data/logs", exist_ok=True)

        file1_lines = await load_file_lines_async(str("data/transcript.txt"))
        file2_lines = await load_file_lines_async(str("data/logs/chat_log.txt"))

        base_ts, _, base_line = find_line_by_pattern(file2_lines, pattern)

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

async def load_file_lines_async(file_path):
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        return await f.readlines()

def find_line_by_pattern(lines, pattern):
    regex = re.compile(pattern)
    for ts, content, full_line in lines:
        if regex.search(full_line):
            return ts, content, full_line
    return None, None, None

def find_closest_line(target_ts, lines):
    # Assuming lines sorted by timestamp ascending
    timestamps = [ts for ts, _, _ in lines]
    pos = bisect.bisect_left(timestamps, target_ts)
    
    candidates = []
    if pos < len(lines):
        candidates.append(lines[pos])
    if pos > 0:
        candidates.append(lines[pos - 1])
    
    # Pick the closest
    closest = min(candidates, key=lambda x: abs(x[0] - target_ts))
    return closest
    
def parse_timestamp(line, timestamp_re = re.compile(r"\[(.*?)\]")):
    match = timestamp_re.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")   

def split_line(line):
    ts = parse_timestamp(line)
    content = line[line.find("]")+1:].strip() if ts else None
    return ts, content, line