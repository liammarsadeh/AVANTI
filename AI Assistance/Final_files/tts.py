import sounddevice as sd
import numpy as np
from piper import PiperVoice


voice = PiperVoice.load(
    r"C:\Users\User\OneDrive\Desktop\en_US-lessac-medium.onnx"
)


def speak(text):

    audio = []

    for chunk in voice.synthesize(text):
        audio.append(chunk.audio_int16_array)

    audio = np.concatenate(audio)

    sd.play(
        audio,
        samplerate=voice.config.sample_rate
    )
    sd.wait()
