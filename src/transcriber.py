# -*- coding: utf-8 -*-
# Audio-related imports
import numpy as np
import sounddevice as sd # type: ignore
import webrtcvad    # type: ignore
from librosa import resample as librosa_resample # type: ignore

# Misc imports
import os, logging
import warnings
from dotenv import load_dotenv
from datetime import datetime, timedelta # type: ignore

# Thread-related imports
from queue import Queue, Full
from threading import Thread

# Static typing imports
from numpy import typing as npt
from typing import List, Tuple, Dict, Generator, Any, Union, Optional # type: ignore
from dataclasses import dataclass

# Transcription-specific imports
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline # type: ignore


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

# Set up custom types
AudioConfig = Dict[str, int | str | float]
MiscConfig = Dict[str, str]

@dataclass
class Frame:
    """Represents a "frame" of audio data."""
    data: bytes
    timestamp: Union[datetime, float]
    duration: float

class Transcriber:
    def __init__(self):
        """Initialize the Transcriber instance.

        Loads environment variables, sets up audio and miscellaneous configurations,
        initializes the audio buffer, and configures logging to write to a file.
        """
        
        # Load environment values
        load_dotenv()
        
        # === Audio configuration global variables ===
        self.AUDIO_CONFIG: AudioConfig = {
            "ORIG_RATE": 48000,
            "TARGET_RATE" : 16000,
            "CHANNELS": 2,
            "FORMAT": 'int16',
            "CHUNK": 960,
            "INPUT_DEVICE_INDEX": 0,
        }
        
        # === Miscellanious config ===
        self.CONFIG: MiscConfig = {
            "LOG_DIR": "data/logs",
        }
        
        # === Configure logging ===
        os.makedirs(self.CONFIG["LOG_DIR"], exist_ok=True)
        self.CONFIG["LOG_FILE"] = os.path.join(self.CONFIG["LOG_DIR"], "program.log")
        
        logging.basicConfig(
            level=logging.DEBUG,
            filename=self.CONFIG["LOG_FILE"],
            encoding="utf-8",
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # === Define a baseline timestamp ===
        self.recording_start_time: datetime = datetime.now()
        
        # === Set up Queues ===
        self.audio_queue: Queue[bytes] = Queue(maxsize=400)
        self.voiced_queue: Queue[Tuple[npt.NDArray[np.float32], float, datetime]] = Queue(maxsize=400)
        self.transcription_queue: Queue[Tuple[datetime, str, float]] = Queue(maxsize=200)
        
        # === Set up Faster Whisper === 
        self.whisper_model = WhisperModel(
            model_size_or_path="distil-large-v3", 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        self.batched_model = BatchedInferencePipeline(self.whisper_model)
        
        # === Set up VAD ===
        self.vad = webrtcvad.Vad(2)
        
    # === Define the main Threads and processes ===
    def start(self):
        Thread(target=self._record_audio, daemon=True).start()
        Thread(target=self._transcribe_audio, daemon=True).start()
        Thread(target=self._apply_vad, daemon=True).start()
    
    # === Define function to convert int16 PCM to float32
    def bytes_to_float32(self, mono_int16_bytes: bytes) -> npt.NDArray[np.float32]:
        int16_array = np.frombuffer(mono_int16_bytes, dtype=np.int16)
        return int16_array.astype(np.float32) / 32768.0
    
    # === Set INPUT_DEVICE_INDEX to the proper one ===
    def set_input_device(self) -> int:
        """Automatically select the audio input device index.

        Scans available audio devices and returns the index of the device that has
        exactly 2 input channels and contains 'Voicemeeter Out B2' in its name.
        Returns 0 if no matching device is found.

        Returns:
            int: The index of the selected input device.
        """
        devices: (sd.DeviceList | dict[str, Any]) = sd.query_devices() # type: ignore
        
        for idx, device in enumerate(devices):
            device_dict = dict(device)
            if device_dict["max_input_channels"] == 2 and "Voicemeeter Out B2" in device_dict["name"]:
                return idx
        return 0
        
    # === Define audio_callback for sounddevice ===
    def audio_callback(self, indata: npt.NDArray[np.int16], frames: int, time: Any, status: Any):
        """Callback function for sounddevice InputStream.

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
        stereo_int16_48k: npt.NDArray[np.int16] = indata.copy() # type: ignore
        
        # Convert int16 stereo -> float32 mono (left channel)
        mono_float_48k: npt.NDArray[np.float32] = stereo_int16_48k[:, 0].astype(np.float32) / 32768.0
        
        # Downsample to 16KHz
        mono_16k: npt.NDArray[np.float32] = librosa_resample( # type: ignore
            mono_float_48k, 
            orig_sr=float(self.AUDIO_CONFIG["ORIG_RATE"]), 
            target_sr=float(self.AUDIO_CONFIG["TARGET_RATE"])
        )
        
        # Convert float32 mono 16kHz -> int16 for VAD
        mono_16k_int16: bytes = (mono_16k * 32768).astype(np.int16).tobytes()
        
        # Append to buffer
        try:
            self.audio_queue.put(mono_16k_int16)
        except Full:
            logging.warning("audio_queue full. Dropping audio chunk.")
        except Exception as e:
            logging.error(f"Failed putting audio data into audio_queue\nError: [{e}]")
            
    # === Split frames for VAD ===
    def frame_generator(self, frame_duration_ms: int, audio: bytes, sample_rate: int) -> Generator[Frame, None, None]:
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration. Yield is just like return, just it doesn't
        restart, rather it pauses and continues from where it left yielded a result. The reason
        for not using return is that it would exit the thread.
        
        Which in our case means a Frame is a 20ms long segment, with timestamp and size in
        bytes.
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
            callback=self.audio_callback # type: ignore
        )
        
        # Start the input stream
        stream.start()
        logging.info("🎙️ Recording started...")
        
    # === VAD Thread ===
    def _apply_vad(self):
        """Accumulate tiny frames into usable audio clips
        
        This function calls frame_generator() very frequently, concatenates the pieces
        and assemble a whole few seconds long audio chunk, puts it into the voiced_queue,
        since frame_generator() only yields voiced audio segments.
        
        
        """
        silence_duration = 0.0
        silence_threshold = 1.5  # Seconds of silence to consider as break
        frame_duration_s  = 0.02  # 20ms frames
        voiced_frames: List[bytes] = []
        first_frame_ts = None
        
        while True:
            if not self.audio_queue.empty():
                int16_audio: bytes = self.audio_queue.get()
                
                # Slice the raw bytes into 20ms frames for VAD
                for frame in self.frame_generator(20, int16_audio, int(self.AUDIO_CONFIG["TARGET_RATE"])):
                    if self.vad.is_speech(frame.data, sample_rate=self.AUDIO_CONFIG["TARGET_RATE"]): # type: ignore
                        voiced_frames.append(frame.data)
                        
                        if first_frame_ts is None:
                            first_frame_ts = frame.timestamp
                        
                        silence_duration = 0 # resets on speech
                        
                    else:
                        if voiced_frames:
                            silence_duration += frame_duration_s
                            
                            if silence_duration >= silence_threshold:
                                # Flush voiced_frames as one chunk
                                voiced_audio = b''.join(voiced_frames)
                                
                                # Calculate clip duration
                                duration = len(voiced_frames) * frame_duration_s
                                
                                if first_frame_ts is not None:
                                    # Define start of audio chunk
                                    chunk_start_dt = self.recording_start_time + timedelta(seconds=first_frame_ts) # type: ignore
                                    
                                    try:
                                        self.voiced_queue.put((self.bytes_to_float32(voiced_audio), duration, chunk_start_dt), block=False)
                                        
                                    except Full:
                                        logging.warning("Failed putting audio into voiced_queue, queue is full.")
                                        
                                    except Exception as e:
                                        logging.error(f"Unexpected error while putting to voiced_queue: {e}")
                                    
                                    finally:
                                        voiced_frames.clear()
                                        silence_duration = 0
                                        first_frame_ts = None
                                else:
                                    logging.warning("first_frame_ts was None, skipping voiced chunk")
                                    voiced_frames.clear()
                                    silence_duration = 0
                                    first_frame_ts = None
        
    # === Transcriber Thread === 
    def _transcribe_audio(self):
        """Continuously processes audio chunks from the audio queue.

        Retrieves audio data with timestamps from `audio_queue`, processes it 
        (e.g. batching, formatting, or sending to a speech-to-text model), and
        places the results into the `transcription_queue`.

        This runs on a separate thread to ensure recording is not blocked by
        the transcription process.
        """
        
        while True:
            if not self.voiced_queue.empty():
                # Read audio queue for timestamp and buffer
                audio_chunk, duration, chunk_start_time = self.voiced_queue.get()
                audio_chunk: npt.NDArray[np.float32]
                
                # Start transcription
                try:
                    segments, _ = self.batched_model.transcribe( # type: ignore
                        audio=audio_chunk,                        
                        beam_size=4,
                        batch_size=8
                    )
                    
                    for seg in segments:
                        if seg.text.strip():
                            text = seg.text.strip()
                            
                            if text:
                                # Put timestamp and text in the transcription queue
                                try:
                                    self.transcription_queue.put((chunk_start_time, text, duration), block=False)
                                    with open("data/transcription.txt", "at", encoding="utf-8", errors="ignore") as tc:
                                        tc.write(f"[{chunk_start_time.strftime("%Y-%m-%d %H:%M:%S")}] Transcription: {text}\n")
                                
                                except Full:
                                    logging.warning("transcription_queue full, dropping transcription")
                    
                except Exception as e:
                    logging.error(f"Error during transcription:\n{e}")
                    
    def get_latest_transcription(self) -> Optional[Tuple[datetime, str, float]]:
        """
        Retrieves the most recent transcription from the transcription_queue.
        If no transcription is available, returns None.
        """
        
        latest = None
        try:
            while True:
                # Keep popping until queue is empty
                latest = self.transcription_queue.get_nowait()
        except Exception:
            pass
        
        return latest