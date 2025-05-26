import pyaudio, wave


CHUNK = 960
RATE = 48000
FORMAT = pyaudio.paInt16
CHANNELS = 2
RECORD_SECONDS = 60

def set_interface_index() -> int:
    p = pyaudio.PyAudio()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "Voicemeeter Out B2" in str(info['name']) and info['maxInputChannels'] == 2:
            return i
    p.terminate()
    return 0

p = pyaudio.PyAudio()
INDEX = set_interface_index()

stream = p.open(
    rate=RATE, channels=CHANNELS, format=FORMAT,
    input_device_index=INDEX, input=True, 
    frames_per_buffer=CHUNK
)

frames = []

# Store data in chunks for 3 seconds
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

with wave.open("data/speaker_samples/filian_sample.wav", "wb") as wf:
    wf.setframerate(RATE)
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.writeframes(b''.join(frames))
    
p.terminate()
stream.stop_stream()
stream.close()