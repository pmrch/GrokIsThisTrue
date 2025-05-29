# -*- coding: utf-8 -*-
# Audio-related imports
import numpy as np
import librosa
import sounddevice as sd

# Misc imports
from dotenv import load_dotenv
import os, logging
import warnings

# Partial imports
from queue import Queue


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

class Transcriber:
    def __init__(self):
        # Load environment values
        load_dotenv()
        
        # Audio configuration global variables
        self.AUDIO_CONFIG = {
            "ORIG_RATE": 48000,
            "TARGET_RATE" : 16000,
            "CHANNELS": 2,
            "FORMAT": 'int16',
            "CHUNK": 960,
            "INPUT_DEVICE_INDEX": 0,
        }
        
        # Miscellanious config
        self.CONFIG = {
            "LOG_DIR": "data/logs",
        }
        
        # Declare global audio buffer
        self.audio_buffer = np.empty((0,), dtype=np.float32)
        
        # Configure logging
        os.makedirs(self.CONFIG["LOG_DIR"], exist_ok=True)
        self.CONFIG["LOG_FILE"] = os.path.join(self.CONFIG["LOG_DIR"], "program.log")
        
        logging.basicConfig(
            level=logging.DEBUG,
            filename=self.CONFIG["LOG_FILE"],
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
    # Set INPUT_DEVICE_INDEX to the proper one
    def set_input_device(self) -> int:
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            device_dict = dict(device)
            if device_dict["max_input_channels"] == 2 and "Voicemeeter Out B2" in device_dict["name"]:
                return idx
        return 0
        
    # Define audio_callback for sounddevice
    def audio_callback(self, indata, frames, time, status):            
        # indata shape: (frames, channels), int16
        stereo_int16_48k = indata.copy()
        
        # Convert int16 stereo -> float32 mono (left channel)
        mono_float_48k = stereo_int16_48k[:, 0].astype(np.float32) / 32768.0
        
        #  Downsample to 16KHz
        mono_16k = librosa.resample(
            mono_float_48k, 
            orig_sr=self.AUDIO_CONFIG["ORIG_RATE"], 
            target_sr=self.AUDIO_CONFIG["TARGET_RATE"]
        )
        
        # Append to buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, mono_16k))
        
    def _record_audio(self):
        
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
        logging.info("Input stream successfully started, recording...")