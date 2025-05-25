import pyaudio, torch, librosa
import threading, queue, difflib, webrtcvad
import os, numpy as np
from faster_whisper import WhisperModel

from datetime import datetime


class Transcriber:
    def __init__(self):
        # === Audio Config ===
        self.CHUNK = 960
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 48000
        self.RECORD_SECONDS = 15
        self.INPUT_DEVICE_INDEX = self.set_interface_index()

        # === Queues and File Setup ===
        os.makedirs("recording", exist_ok = True)
        self.WAVE_OUTPUT_FILENAME = "recording/temp_audio.wav"
        
        self.audio_queue = queue.Queue() # Store (timestamp. transcription)
        self.transcription_queue = queue.Queue()
        
        # === Whisper Initialization ===
        self.whisper_model = WhisperModel("distil-large-v3", device="cuda" if torch.cuda.is_available() else "cpu", 
                                          compute_type="float16" if torch.cuda.is_available() else "int8")
        
        # Set up deduplication
        self.prev_transcription = ""
        
        # Allow overlaps
        self.OVERLAP_SECONDS = 2  # seconds of overlap
        self.last_frames = []
        
        # === Set up VAD ===
        self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3, 2 is balanced
        

    # === Device Selection ===
    def set_interface_index(self) -> int:
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "Voicemeeter Out B2" in info['name'] and info['maxInputChannels'] == 2:
                return i
            
        p.terminate()
        raise RuntimeError("Voicemeeter device not found")

    def dedupe_overlap(self, prev_text, new_text) -> str:
        matcher = difflib.SequenceMatcher(None, prev_text, new_text)
        match = matcher.find_longest_match(0, len(prev_text), 0, len(new_text))
        
        if match.size > 20 and match.b + match.size < len(new_text):
            return new_text[match.b + match.size:].strip()
        return new_text
        
    def vad_detect(self, audio_int16, sample_rate=16000, frame_duration_ms=20):
        n = int(sample_rate * frame_duration_ms / 1000)
        
        offset = 0
        while offset + n <= len(audio_int16):
            frame = audio_int16[offset:offset + n]
            frame_bytes = frame.tobytes()
            
            if self.vad.is_speech(frame_bytes, sample_rate):
                return True
            offset += n
        
        return False
        
    def fix_punctuation(self, text):
        import re
        
        text = text.strip()

        # Fix broken interjections like 'Ah.', 'Friggin.' -> 'Ah!' or 'Friggin...'
        text = re.sub(r"\b(Ah|Oh|Friggin|Ugh)\.(\s|$)", r"\1... ", text)

        # Fix 'Cause' at beginning of sentence -> 'Because'
        text = re.sub(r"^(Cause|cause)\b", "Because", text)

        # Collapse repeated words like "drip. drip drip"
        text = re.sub(r"\b(\w+)\.\s+\1\s+\1\b", r"\1, \1, \1", text)

        # Capitalize after punctuation if needed
        text = re.sub(r"([.!?])\s+([a-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", text)

        # Fix random "I." becoming ellipses or misread
        text = re.sub(r"\b[Ii]\.\s+", "I... ", text)

        # Clean up filler exclamations and emphasize naturally
        text = re.sub(r"\b(dude|wow|yeah|okay|sure)\b", lambda m: m.group(1).capitalize(), text, flags=re.IGNORECASE)
        text = re.sub(r"\b(Dude|Wow|Yeah|Okay|Sure)[.!]*", r"\1!", text)

        # Remove unnecessary spaces before punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)

        # Remove excess spacing
        text = re.sub(r"\s+", " ", text)

        # Sponsor formatting (non-intrusive way)
        text = re.sub(r"(check out.*(description|sponsor).*)", "[sponsor segment]", text, flags=re.IGNORECASE)
        
        # Fix awkward 'I' edge cases
        text = re.sub(r"\b[Ii]\.(\s+|$)", "I... ", text)

        # End with punctuation if missing
        if text and text[-1] not in ".!?":
            text += "."
            
        if len(text.split()) <= 2:
            text = f"[fragment] {text}"

        return text
    
    def clean_and_merge_transcript_lines(self, lines):
        merged = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect timestamps
            if len(line) >= 10 and line[0] == "[" and line[9] == "]":
                # Line starts with a timestamp
                line_text = line[10:].strip()

                if buffer:
                    merged.append(self.fix_punctuation(buffer.strip()))
                buffer = line_text
            else:
                # It's a continuation
                if line[0].islower() or line[0] in ",.;":
                    buffer += " " + line
                else:
                    # Unexpected new sentence start – still merge conservatively
                    buffer += " " + line

        if buffer:
            merged.append(self.fix_punctuation(buffer.strip()))
            
        # Remove adjacent duplicates
        deduped = [merged[0]] if merged else []
        for line in merged[:1]:
            if line != deduped[-1]:
                deduped.append(line)
                
        return deduped
    
    def start(self):
        threading.Thread(target=self._record_audio, daemon=True).start()
        threading.Thread(target=self._transcribe_audio, daemon=True).start()  

    # === Recorder Thread ===
    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, 
            input=True, frames_per_buffer=self.CHUNK, 
            input_device_index=self.INPUT_DEVICE_INDEX
        )
        
        print("🎙️ Recording started...")
        
        try:
            while True:
                frames = []
                
                for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                current_chunk = b''.join(frames)
                if self.last_frames:
                    # Join last few seconds with current
                    audio_data = b''.join(self.last_frames) + current_chunk
                else:
                    audio_data = current_chunk
                    
                # Update the last_frames for next round
                overlap_chunks = int(self.RATE / self.CHUNK * self.OVERLAP_SECONDS)
                self.last_frames = frames[-overlap_chunks:]
                
                timestamp = datetime.now()
                self.audio_queue.put((timestamp, audio_data))
            
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    # === Transcriber Thread ===
    def _transcribe_audio(self):
        while True:
            if not self.audio_queue.empty():
                timestamp, audio_data = self.audio_queue.get()
                
                # Convert stereo to mono
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                if self.CHANNELS == 2:
                    audio_np = audio_np.reshape(-1, 2).mean(axis=1)
                    
                # Normalize and resample to 16KHz
                audio_float = audio_np.astype(np.float32) / 32768.0
                audio_resampled = librosa.resample(audio_float, orig_sr = self.RATE, target_sr = 16000.0)
                
                # Prepare for VAD
                audio_int16_16k = (audio_resampled * 32768).astype(np.int16)
                
                if not self.vad_detect(audio_int16_16k, sample_rate=16000):
                    # No speech detected, skip transcription
                    continue
                
                # Whisper expects numpy array or torch tensor
                try:
                    segments, _ = self.whisper_model.transcribe(audio_resampled, vad_filter=True)
                    
                    raw_lines = [f"[{timestamp.strftime('%H:%M:%S')}] {seg.text.strip()}" for seg in segments]
                    cleaned_text = self.clean_and_merge_transcript_lines(raw_lines)
                    new_text = " ".join([line for line in cleaned_text]).strip()
                    
                    if new_text:
                        # Deduplicate overlapping text
                        deduped_text = self.dedupe_overlap(self.prev_transcription, new_text)
                        self.prev_transcription = new_text
                        
                        if deduped_text.strip():
                            self.transcription_queue.put((timestamp, deduped_text))
                            os.makedirs("data", exist_ok=True)
                        
                            with open("data/transcription.txt", "at", encoding = 'utf-8', errors = 'ignore') as file:
                                file.write(f"[{timestamp.strftime('%H:%M:%S')}] {deduped_text}\n")
                        
                except Exception as e:
                    print(f"Whisper error \n{e}")

    # Get latest transcription within time window
    def get_latest_transcription(self, max_age_seconds=30):
        current_time = datetime.now()
        while not self.transcription_queue.empty():
            ts, text = self.transcription_queue.queue[-1]
            if (current_time - ts).total_seconds() <= max_age_seconds:
                return text
            self.transcription_queue.get()
        return ""