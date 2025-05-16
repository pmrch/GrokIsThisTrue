import pyaudio

p = pyaudio.PyAudio()
device_index = 78  # your device

for rate in [44100, 48000, 32000, 22050, 16000]:
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=2,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=1024)
        stream.close()
        print(f"Sample rate {rate} supported")
    except Exception as e:
        print(f"Sample rate {rate} NOT supported: {e}")

p.terminate()