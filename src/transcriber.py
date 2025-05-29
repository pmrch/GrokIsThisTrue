# -*- coding: utf-8 -*-
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

import pyaudio, librosa
import threading, queue, difflib, os, re
import warnings
from datetime import datetime, timedelta
from time import sleep

from faster_whisper import WhisperModel, BatchedInferencePipeline
from dotenv import load_dotenv

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

class Transcriber:
    def __init__(self):
        # Load Environment variables
        load_dotenv()
        
        # Fix warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
        huggingface_token = os.getenv("HuggingFaceToken")
        
        if not huggingface_token:
            raise ValueError("HuggingFaceToken not found in .env file.")
        
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
        self.whisper_model = WhisperModel(
            "distil-large-v3",  # Upgrade from distil-large-v3 to large-v3
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"  # Use float16 on GPU for better accuracy
        )
        self.batched_model = BatchedInferencePipeline(model=self.whisper_model)
        
        # === Set up deduplication ===
        self.prev_transcription = ""
        
        # === Allow overlaps ===
        self.OVERLAP_SECONDS = 5  # seconds of overlap
        self.last_frames = []
                        
    # === Device Selection ===
    def set_interface_index(self) -> int:
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "Voicemeeter Out B2" in str(info['name']) and info['maxInputChannels'] == 2:
                return i
            
        p.terminate()
        raise RuntimeError("Voicemeeter device not found")

    def dedupe_overlap(self, prev_text: str, new_text: str) -> str:
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
        if match.size > 5:  # Simplified check without speaker comparison
            return ' '.join(new_text.split()[match.b + match.size:]).strip()
        return new_text
        
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
                chunk_timestamp, audio_data = self.audio_queue.get()
                
                # Convert stereo to mono
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                if self.CHANNELS == 2:
                    audio_np = audio_np.reshape(-1, 2)[:, 0]
                    
                # Normalize audio volume to avoid clipping and resample to 16KHz
                audio_float = audio_np.astype(np.float32) / 32768.0
                audio_float = np.clip(audio_float / np.max(np.abs(audio_float) + 1e-10), -1.0, 1.0)
                audio_resampled = librosa.resample(audio_float, orig_sr=self.RATE, target_sr=16000.0)
                
                try:
                    segments, _ = self.batched_model.transcribe(
                        audio_resampled, 
                        vad_filter=True, 
                        batch_size=4,
                        vad_parameters=dict(threshold=0.8, min_silence_duration_ms=500)
                    )
                    
                    # Align transcription with segments
                    raw_lines = []
                    seen_texts = set()  # Track unique texts to avoid duplicates
                    
                    for seg in segments:
                        if seg.text.strip():
                            text = seg.text.strip()
                            if text in seen_texts:
                                continue
                            seen_texts.add(text)
                            
                            # Calculate segment-specific timestamp
                            segment_time = chunk_timestamp + timedelta(seconds=seg.start)
                            
                            deduped_text = self.dedupe_overlap(self.prev_transcription, text)
                            if deduped_text.strip():
                                self.prev_transcription = text
                                formatted_line = f"[{segment_time.strftime('%Y-%m-%d %H:%M:%S')}] {deduped_text}"
                                raw_lines.append((segment_time, formatted_line))
                                
                    if raw_lines:
                        # Sort by timestamp before writing
                        raw_lines.sort(key=lambda x: x[0])
                        formatted_lines = [line for _, line in raw_lines]
                        
                        self.transcription_queue.put((chunk_timestamp, "\n".join(formatted_lines)))
                        os.makedirs("data", exist_ok=True)
                        
                        with open("data/transcription.txt", "at", encoding="utf-8", errors="ignore") as file:
                            file.write("\n".join(formatted_lines) + "\n")
                            
                except RuntimeError as e:
                    print(f"Transcription error: \n{e}")
                    if "CUDA out of memory" in str(e):
                        print("CUDA memory error. Try reducing batch_size or using a smaller model.")
                        
                except Exception as e:
                    print(f"Unexpected error \n{e}")
                    
            else:
                # If queue is empty, sleep to prevent CPU spinning
                sleep(0.25)

    # Get latest transcription within time window
    def get_latest_transcription(self, max_age_seconds=30):
        current_time = datetime.now()
        while not self.transcription_queue.empty():
            ts, text = self.transcription_queue.queue[-1]
            if (current_time - ts).total_seconds() <= max_age_seconds:
                return text
            self.transcription_queue.get()
        return ""


