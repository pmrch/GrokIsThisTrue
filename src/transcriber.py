# -*- coding: utf-8 -*-
# Audio-related imports
import numpy as np
import librosa
import sounddevice as sd
import webrtcvad

# Misc imports
import os, logging
import warnings
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Thread-related imports
from queue import Queue, Full
from threading import Thread

# Static typing imports
from numpy import typing as npt

# Transcription-specific imports
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

class Transcriber:
    def __init__(self):
        """
        Initialize the Transcriber instance.

        Loads environment variables, sets up audio and miscellaneous configurations,
        initializes the audio buffer, and configures logging to write to a file.
        """
        
        # Load environment values
        load_dotenv()
        
        # === Audio configuration global variables ===
        self.AUDIO_CONFIG = {
            "ORIG_RATE": 48000,
            "TARGET_RATE" : 16000,
            "CHANNELS": 2,
            "FORMAT": 'int16',
            "CHUNK": 960,
            "INPUT_DEVICE_INDEX": 0,
        }
        
        # === Miscellanious config ===
        self.CONFIG = {
            "LOG_DIR": "data/logs",
        }
        
        # Configure logging
        os.makedirs(self.CONFIG["LOG_DIR"], exist_ok=True)
        self.CONFIG["LOG_FILE"] = os.path.join(self.CONFIG["LOG_DIR"], "program.log")
        
        logging.basicConfig(
            level=logging.DEBUG,
            filename=self.CONFIG["LOG_FILE"],
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # === Set up Queues ===
        self.audio_queue = Queue(maxsize=100)
        self.transcription_queue = Queue(maxsize=100)
        
        # === Set up Faster Whisper === 
        self.whisper_model = WhisperModel(
            model_size_or_path="distil-large-v3", 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        self.batched_model = BatchedInferencePipeline(self.whisper_model)
        
    # === Define the main Threads and processes ===
    def start(self):
        Thread(target=self._record_audio, daemon=True).start()
        Thread(target=self._transcribe_audio, daemon=True).start()
        Thread(target=self._apply_vad, daemon=True).start()
    
    # === Set INPUT_DEVICE_INDEX to the proper one ===
    def set_input_device(self) -> int:
        """
        Automatically select the audio input device index.

        Scans available audio devices and returns the index of the device that has
        exactly 2 input channels and contains 'Voicemeeter Out B2' in its name.
        Returns 0 if no matching device is found.

        Returns:
            int: The index of the selected input device.
        """
        
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            device_dict = dict(device)
            if device_dict["max_input_channels"] == 2 and "Voicemeeter Out B2" in device_dict["name"]:
                return idx
        return 0
        
    # === Define audio_callback for sounddevice ===
    def audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """
        Callback function for sounddevice InputStream.

        Called automatically when new audio data is available.

        Args:
            indata (numpy.ndarray): Audio input data of shape (frames, channels), dtype=int16.
            frames (int): Number of frames.
            time (CData): Time information (ignored).
            status (CallbackFlags): Status of the input stream.

        Process:
            - Copies stereo int16 audio chunk.
            - Converts left channel to float32 mono normalized [-1, 1].
            - Downsamples from 48kHz to 16kHz.
            - Appends downsampled audio to the global audio buffer.
        """
              
        # indata shape: (frames, channels), int16
        stereo_int16_48k: npt.NDArray[np.int16] = indata.copy()
        
        # Convert int16 stereo -> float32 mono (left channel)
        mono_float_48k: npt.NDArray[np.float32] = stereo_int16_48k[:, 0].astype(np.float32) / 32768.0
        
        # Downsample to 16KHz
        mono_16k: npt.NDArray[np.float32] = librosa.resample(
            mono_float_48k, 
            orig_sr=self.AUDIO_CONFIG["ORIG_RATE"], 
            target_sr=self.AUDIO_CONFIG["TARGET_RATE"]
        )
        
        # Convert float32 mono 16kHz -> int16 for VAD
        mono_16k_int16: bytes = (mono_16k * 32768).astype(np.int16).tobytes()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Append to buffer
        try:
            self.audio_queue.put((timestamp, mono_16k, mono_16k_int16))
        except Full:
            logging.warning("Audio queue full. Dropping audio chunk.")
            
    # === Split frames for VAD ===
    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        
        # Calculate how many bytes correspond to the frame duration
        # 16-bit PCM = 2 bytes per sample
        frame_byte_size = int(sample_rate * (frame_duration_ms / 1000.0)) * 2
        
        offset = 0
        timestamp = 0.0
        frame_duration = frame_duration_ms / 1000.0
        
        while offset + frame_byte_size < len(audio):
            yield Frame(audio[offset:offset + frame_byte_size], timestamp, frame_duration)
            timestamp += frame_duration
            offset += frame_byte_size
        
        
    # === Recorder Thread ===
    def _record_audio(self):
        """
        Start recording audio from the selected input device.

        Opens an input stream with the configured samplerate, channels, and chunk size.
        Uses audio_callback to handle incoming audio data.
        Starts the stream and logs that recording has started.
        """
        
        # Create the input stream
        stream = sd.InputStream(
            samplerate=self.AUDIO_CONFIG["ORIG_RATE"],
            channels=self.AUDIO_CONFIG["CHANNELS"],
            dtype=self.AUDIO_CONFIG["FORMAT"],
            blocksize=self.AUDIO_CONFIG["CHUNK"],
            callback=self.audio_callback
        )
        
        # Start the input stream
        stream.start()
        logging.info("🎙️ Recording started...")
        
    # === VAD Thread ===
    def _apply_vad(self):
        while True:
            if not self.audio_queue.empty():
                ts, float_audio, int16_audio = self.audio_queue.get()
                
                # Slice the raw bytes into 20ms frames for VAD
        
    # === Transcriber Thread === 
    def _transcribe_audio(self):
        """
        Continuously processes audio chunks from the audio queue.

        Retrieves audio data with timestamps from `audio_queue`, processes it 
        (e.g. batching, formatting, or sending to a speech-to-text model), and
        places the results into the `transcription_queue`.

        This runs on a separate thread to ensure recording is not blocked by
        the transcription process.
        """
        
        while True:
            if not self.audio_queue.empty():
                # Read audio queue for timestamp and buffer
                timestamp, audio_chunk = self.audio_queue.get()
                
                # Start transcription
                try:
                    segments, _ = self.batched_model.transcribe(
                        audio=audio_chunk,
                        beam_size=4,
                        batch_size=4
                    )
                    
                except Exception as e:
                    logging.error(f"Error during transcription:\n{e}")