# -*- coding: utf-8 -*-
import pyaudio, librosa, numpy as np
import torch, torch.nn.functional as F
import threading, queue, difflib, os, re
import warnings, speechbrain as sb

from typing import Dict, Optional, Union, List, Tuple
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from datetime import datetime
from speechbrain.inference import EncoderClassifier  # Updated import path
from speechbrain.utils.fetching import LocalStrategy

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
        
        sb.utils.fetching.fetch_strategy = LocalStrategy.COPY # type: ignore
        
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
        
        # === Diarization Initialization ===
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=str(os.getenv("HuggingFaceToken")))
        self.prev_transcription_speaker = None
        
        # === Set up deduplication ===
        self.prev_transcription = ""
        
        # === Allow overlaps ===
        self.OVERLAP_SECONDS = 5  # seconds of overlap
        self.last_frames = []
        
        # === Speaker Recognition Initialization ===
        try:
            self.speaker_recognizer = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        except Exception as e:
            print(f"Failed to initialize speaker recognition: {e}")
            self.speaker_recognizer = None
        
        # === Speaker Embeddings Storage ===
        self.speaker_embeddings: Dict[str, Optional[torch.Tensor]] = {
            "Neuro": None,
            "Filian": None,
            "Vedal": None
        }
        
        # Try to load speaker samples if they exist
        self.load_speaker_samples()
        
    def load_speaker_samples(self) -> None:
        """Load and register speaker samples from the data/speaker_samples directory"""
        samples_dir = os.path.join("data", "speaker_samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        speakers = {
            "Neuro": "neuro_sample.wav",
            "Filian": "filian_sample.wav",
            "Vedal": "vedal_sample.wav"
        }
        
        for speaker, filename in speakers.items():
            audio_path = os.path.join(samples_dir, filename)
            if os.path.exists(audio_path):
                try:
                    # Load audio file and convert to 16kHz mono
                    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                    self.register_speaker(speaker, audio)
                    
                except Exception as e:
                    print(f"Failed to load speaker sample for {speaker}: {e}")
                    
    # === Device Selection ===
    def set_interface_index(self) -> int:
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "Voicemeeter Out B2" in str(info['name']) and info['maxInputChannels'] == 2:
                return i
            
        p.terminate()
        raise RuntimeError("Voicemeeter device not found")

    def dedupe_overlap(self, prev_text: str, new_text: str, prev_speaker=None, new_speaker=None) -> str:
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
        if match.size > 5 and prev_speaker == new_speaker:  # Lower threshold for word-level match
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
    
    def clean_and_merge_transcript_lines(self, lines, diarization_results):
        merged = []
        buffer = ""
        current_speaker = None

        for line, (start, end, speaker) in zip(lines, diarization_results):
            line = line.strip()
            if not line:
                continue

            # Extract timestamp and text
            if len(line) >= 10 and line[0] == "[" and line[9] == "]":
                if buffer:
                    merged.append((self.fix_punctuation(buffer.strip()), current_speaker))
                buffer = line[10:].strip()
                current_speaker = speaker
            else:
                buffer += " " + line

        if buffer:
            merged.append((self.fix_punctuation(buffer.strip()), current_speaker))

        # Format lines with speaker names
        formatted = []
        for text, speaker in merged:
            if text:
                speaker_name = f"[{speaker}]" if speaker and speaker != "Unknown" else ""
                formatted.append((f"{speaker_name} {text}".strip(), speaker))
            
        return formatted
    
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
                audio_float = np.clip(audio_float / np.max(np.abs(audio_float) + 1e-10), -1.0, 1.0)
                audio_resampled = librosa.resample(audio_float, orig_sr=self.RATE, target_sr=16000.0)
                
                # Perform diarization
                try:
                    diarization = self.diarization_pipeline({
                        "waveform": torch.tensor(audio_float, dtype=torch.float32).unsqueeze(0), 
                        "sample_rate": self.RATE}, min_speakers=1, max_speakers=5)
                    diarization_results = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
                    
                except Exception as e:
                    print(f"Diarization error: {e}")
                    diarization_results = []
                
                # Whisper expects numpy array or torch tensor
                try:
                    segments, _ = self.batched_model.transcribe(
                        audio_resampled, 
                        vad_filter=True, 
                        batch_size=4,
                        vad_parameters=dict(threshold=0.8, min_silence_duration_ms=500)
                    )
                    
                    # Align transcription with diarization
                    raw_lines = []
                    seen_texts = set()  # Track unique texts to avoid duplicates
                    
                    for seg in segments:
                        if seg.text.strip():
                            # Get audio segment
                            start_sample = int(seg.start * 16000)
                            end_sample = int(seg.end * 16000)
                            segment_audio = audio_resampled[start_sample:end_sample]
                            
                            # Get speaker embedding
                            embedding = self.compute_speaker_embedding(segment_audio)
                            speaker = self.match_speaker(embedding)
                            
                            # Use the matched speaker instead of diarization speaker
                            text = seg.text.strip()
                            if text in seen_texts:
                                continue
                            seen_texts.add(text)
                            
                            raw_lines.append((
                                f"[{timestamp.strftime('%H:%M:%S')}] {text}",
                                (seg.start, seg.end, speaker)
                            ))
                                
                    cleaned_lines = self.clean_and_merge_transcript_lines(
                        [line for line, _ in raw_lines],
                        [di for _, di in raw_lines]
                    )
                    formatted_lines = []
                    deduped_text = str()
                    
                    for text, speaker in cleaned_lines:
                        if text:
                            deduped_text = self.dedupe_overlap(
                                self.prev_transcription,
                                text,
                                prev_speaker=self.prev_transcription_speaker,
                                new_speaker=speaker
                            )
                        if deduped_text.strip():
                            self.prev_transcription = text
                            self.prev_transcription_speaker = speaker
                            speaker_prefix = f"[{speaker}]" if speaker and speaker != "Unknown" else ""
                            formatted_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {speaker_prefix} {deduped_text}"
                            formatted_lines.append(formatted_line)
                            
                    if formatted_lines:
                        self.transcription_queue.put((timestamp, "\n".join(formatted_lines)))
                        os.makedirs("data", exist_ok=True)
                        
                        with open("data/transcription.txt", "at", encoding="utf-8", errors="ignore") as file:
                            file.write("\n".join(formatted_lines) + "\n")
                            
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
    
    def compute_speaker_embedding(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Optional[torch.Tensor]:
        if self.speaker_recognizer is None:
            print("Speaker recognition not available")
            return None
            
        with torch.no_grad():
            try:
                # Ensure audio is long enough to avoid std() warning
                min_samples = int(sample_rate * 0.5)  # At least 0.5 seconds
                if len(audio_array) < min_samples:
                    padded_audio = np.pad(audio_array, (0, min_samples - len(audio_array)))
                    audio_array = padded_audio

                embeddings = self.speaker_recognizer.encode_batch(torch.tensor(audio_array).unsqueeze(0))
                return F.normalize(embeddings, dim=2)
            except Exception as e:
                print(f"Error computing speaker embedding: {e}")
                return None
    
    def match_speaker(self, embedding: Optional[torch.Tensor]) -> str:
        if embedding is None or not any(self.speaker_embeddings.values()):
            return "Unknown"
        
        max_similarity = -1
        matched_speaker = "Unknown"
        
        for speaker, stored_embedding in self.speaker_embeddings.items():
            if stored_embedding is not None:
                similarity = F.cosine_similarity(embedding, stored_embedding)
                if similarity > max_similarity and similarity > 0.75:  # Threshold
                    max_similarity = similarity
                    matched_speaker = speaker
                    
        return matched_speaker
    
    def register_speaker(self, name: str, audio_sample: np.ndarray) -> None:
        """
        Register a known speaker's voice
        
        Args:
            name: Speaker name (must match one of the keys in speaker_embeddings)
            audio_sample: numpy array of audio data (16kHz mono)
        """
        if name not in self.speaker_embeddings:
            raise ValueError(f"Unknown speaker name: {name}")
            
        embedding = self.compute_speaker_embedding(audio_sample)
        self.speaker_embeddings[name] = embedding
        print(f"Registered voice profile for {name}")

