import pyaudio, whisper, torch, librosa
import threading, queue
import os, time, numpy as np
import whisper

from datetime import datetime

# === Device Selection ===
def set_interface_index() -> int:
    p = pyaudio.PyAudio()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "Voicemeeter Out B2" in info['name'] and info['maxInputChannels'] == 2:
            return i
        
    p.terminate()
    raise RuntimeError("Voicemeeter device not found")


# === Audio Config ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
RECORD_SECONDS = 15
INPUT_DEVICE_INDEX = set_interface_index()

os.makedirs("recording", exist_ok = True)
WAVE_OUTPUT_FILENAME = "recording/temp_audio.wav"

# === Initialize Whisper and queue ===
whisper_model = whisper.load_model("medium", device = "cuda" if torch.cuda.is_available() else "cpu") # Fast real-time model
audio_queue = queue.Queue() # Store (timestamp. transcription)
transcription_queue = queue.Queue()

# === Recorder Thread ===
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, 
        input=True, frames_per_buffer=CHUNK, 
        input_device_index = INPUT_DEVICE_INDEX
    )
    
    print("🎙️ Recording started...")
    
    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
    
            audio_data = b''.join(frames)
            timestamp = datetime.now()
            audio_queue.put((timestamp, audio_data))
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# === Transcriber Thread ===
def transcribe_audio():
    while True:
        if not audio_queue.empty():
            timestamp, audio_data = audio_queue.get()
            
            # Convert stereo to mono
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if CHANNELS == 2:
                audio_np = audio_np.reshape(-1, 2).mean(axis=1)
                
            # Normalize and resample to 16KHz
            audio_float = audio_np.astype(np.float32) / 32768.0
            audio_resampled = librosa.resample(audio_float, orig_sr = RATE, target_sr = 16000.0)
            
            # Whisper expects numpy array or torch tensor
            try:
                result = whisper_model.transcribe(audio_resampled, fp16 = torch.cuda.is_available())
                text = result["text"].strip()
                transcription_queue.put((timestamp, text))
                
                # Print immediately or store
                with open("transcription.txt", "at", encoding = 'utf-8', errors = 'ignore') as file:
                    file.write(f"[{timestamp.strftime('%H:%M:%S')}] {text}\n")
                    
            except Exception as e:
                print(f"Whisper error \n{e}")

# Start recording in a thread
def start_transcription():
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=transcribe_audio, daemon=True).start()

# Get latest transcription within time window
def get_latest_transcription(max_age_seconds=30):
    current_time = datetime.now()
    while not audio_queue.empty():
        ts, text = audio_queue.queue[-1]
        if (current_time - ts).total_seconds() <= max_age_seconds:
            return text
        audio_queue.get()
    return ""

# === Main Test Loop ===
if __name__ == "__main__":
    start_transcription()

    last_ts = None
    while True:
        if not transcription_queue.empty():
            ts, text = transcription_queue.queue[-1]
            if last_ts is None or ts > last_ts:
                print(f"🆕 [{ts.strftime('%H:%M:%S')}] New transcription: {text}")
                last_ts = ts
        time.sleep(1)
