import pyaudio
import sys
import wave


# Initialize PyAudio
audio = pyaudio.PyAudio()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 7


input_stream = audio(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    input_device_index = DEVICE_INDEX,
    frames_per_chunk = CHUNK
)

print(RATE, '\n', DEVICE_INDEX)