import pyaudio


def set_interface_index() -> int:
    p = pyaudio.PyAudio()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "Voicemeeter Out B2" in info['name'] and info['maxInputChannels'] == 2:
            return i
    
    p.terminate()