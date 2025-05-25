import re, bisect, aiofiles # type: ignore
from datetime import datetime

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