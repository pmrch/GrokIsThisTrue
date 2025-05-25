import pyaudio, torch, librosa
import threading, queue, difflib, webrtcvad
import os, numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline

from datetime import datetime


class Transcriber:
    def __init__(self):
        # === Audio Config ===
        self.CHUNK = 960
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 48000
        self.RECORD_SECONDS = 30
        self.INPUT_DEVICE_INDEX = self.set_interface_index()

        # === Queues and File Setup ===
        os.makedirs("recording", exist_ok = True)
        self.WAVE_OUTPUT_FILENAME = "recording/temp_audio.wav"
        
        self.audio_queue = queue.Queue() # Store (timestamp. transcription)
        self.transcription_queue = queue.Queue()
        
        # === Whisper Initialization ===
        self.whisper_model = WhisperModel("distil-large-v3", device="cuda" if torch.cuda.is_available() else "cpu", 
                                          compute_type="int8_float16" if torch.cuda.is_available() else "int8")
        self.batched_model = BatchedInferencePipeline(model=self.whisper_model)
        
        # Set up deduplication
        self.prev_transcription = ""
        
        # Allow overlaps
        self.OVERLAP_SECONDS = 5  # seconds of overlap
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
        if not prev_text:
            return new_text
    
        # Estimate overlap region (first 5 seconds of new_text, last 5 seconds of prev_text)
        overlap_duration = self.OVERLAP_SECONDS
        words_per_sec = 2.5  # Approx. words per second in speech
        overlap_words = int(overlap_duration * words_per_sec)
        
        # Split texts into words
        prev_words = prev_text.split()[-overlap_words:]  # Last words of prev_text
        new_words = new_text.split()[:overlap_words]     # First words of new_text
        
        # Compare overlap regions
        matcher = difflib.SequenceMatcher(None, ' '.join(prev_words), ' '.join(new_words))
        match = matcher.find_longest_match(0, len(' '.join(prev_words)), 0, len(' '.join(new_words)))
        
        # If significant overlap, remove the duplicated portion
        if match.size > 5:  # Lower threshold for word-level match
            return ' '.join(new_text.split()[match.b + match.size:]).strip()
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
        
    def fix_punctuation(self, text, style="clean"):
        import re

        text = text.strip()

        if style == "clean":
            # Remove space before punctuation
            text = re.sub(r"\s+([.,!?])", r"\1", text)
            # Collapse all whitespace
            text = re.sub(r"\s+", " ", text)
            # Capitalize after punctuation
            text = re.sub(r"([.!?])\s+([a-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", text)
            # Add final punctuation if needed
            if text and text[-1] not in ".!?":
                text += "."

        elif style == "chaotic":
            # Preserve filler words and stutters like "um", "uh", "ah"
            text = re.sub(r"\b([Uu]m|[Uu]h|[Aa]h)\b", r"\1", text)
            # Keep repeating words like "stop panicking" x5
            text = re.sub(r"\s+", " ", text)
            # Loose capitalization, don't overcorrect
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
            if text and text[-1] not in ".!?":
                text += "..."

        elif style == "minimal":
            # Just remove extra spaces
            text = re.sub(r"\s+", " ", text)

        return text
    
    def clean_and_merge_transcript_lines(self, lines):
        merged = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract timestamp and text
            if len(line) >= 10 and line[0] == "[" and line[9] == "]":
                if buffer:
                    merged.append(self.fix_punctuation(buffer.strip()))
                buffer = line[10:].strip()
            else:
                buffer += " " + line

        if buffer:
            merged.append(self.fix_punctuation(buffer.strip()))
            
        return merged
    
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
                    audio_np = audio_np.reshape(-1, 2)[:, 0]
                    
                # Normalize audio volume to avoid clipping and resample to 16KHz
                audio_float = audio_np.astype(np.float32) / 32768.0
                audio_float = np.clip(audio_float / np.max(np.abs(audio_float)), -1.0, 1.0) # Normalize
                audio_resampled = librosa.resample(audio_float, orig_sr = self.RATE, target_sr = 16000.0)
                
                # Whisper expects numpy array or torch tensor
                try:
                    segments, _ = self.batched_model.transcribe(
                        audio_resampled, vad_filter=True, 
                        batch_size=8,vad_parameters=dict(threshold=0.5)
                    )
                    
                    # Filter segments to exclude overlap region (first 5 seconds of new chunk)
                    overlap_duration = self.OVERLAP_SECONDS
                    raw_lines = [
                        f"[{timestamp.strftime('%H:%M:%S')}] {seg.text.strip()}" 
                        for seg in segments if seg.start >= overlap_duration or seg.end <= self.RECORD_SECONDS
                    ]
                    
                   
                    cleaned_text = self.clean_and_merge_transcript_lines(raw_lines)
                    new_text = " ".join(cleaned_text).strip()
                    
                    if new_text:
                        # Deduplicate overlapping text
                        deduped_text = self.dedupe_overlap(self.prev_transcription, new_text)
                        self.prev_transcription = new_text
                        
                        if deduped_text.strip():
                            self.transcription_queue.put((timestamp, deduped_text))
                            os.makedirs("data", exist_ok=True)
                        
                            with open("data/transcription.txt", "at", encoding = 'utf-8', errors = 'ignore') as file:
                                file.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {deduped_text}\n")
                
                except RuntimeError as e:
                    print(f"Transcription error: \n{e}")
                    if "CUDA out of memory" in str(e):
                        print("CUDA memory error. Try reducing batch_size or using a smaller model.")
                        
                except Exception as e:
                    print(f"Unexpected error \n{e}")

    # Get latest transcription within time window
    def get_latest_transcription(self, max_age_seconds=30):
        current_time = datetime.now()
        while not self.transcription_queue.empty():
            ts, text = self.transcription_queue.queue[-1]
            if (current_time - ts).total_seconds() <= max_age_seconds:
                return text
            self.transcription_queue.get()
        return ""