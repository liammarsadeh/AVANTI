from faster_whisper import WhisperModel
import sounddevice as sd
import re
from LLM import get_completion
from tts import speak
import numpy as np

SAMPLE_RATE = 16000


# Initialize Whisper

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)



# Record audio

def record_audio(duration):

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    return audio.flatten()



# Transcribe

def transcribe_audio(audio):

    segments, info = model.transcribe(
        audio,
        language="en"
    )

    text = ""

    for segment in segments:
        text += segment.text

    text = re.sub(r"[^\w\s]", "", text)

    return text.lower().strip()




CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.001
SILENCE_DURATION = 1


import queue

def listen_until_silence():
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    audio_chunks = []
    silence_time = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                         dtype="float32", blocksize=CHUNK_SIZE,
                         callback=callback):
        print("Listening...")
        while True:
            chunk = q.get().flatten()
            audio_chunks.append(chunk)

            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < SILENCE_THRESHOLD:
                silence_time += CHUNK_SIZE / SAMPLE_RATE
            else:
                silence_time = 0

            if silence_time >= SILENCE_DURATION:
                break

    return np.concatenate(audio_chunks)







active = False

while True:

    if not active:

        print("Waiting for Hey Avanti...")

        audio = record_audio(5)

        text = transcribe_audio(audio)

        print("Heard:", text)

        if "hey avanti" in text:

            print("HEY AVANTI DETECTED!")

            active = True

            speak("How can I help you?")


    else:

        command_audio = listen_until_silence()

        command = transcribe_audio(command_audio)

        print("Command:", command)
        if not command:
            continue

        if "bye avanti" in command:

            print("BYE AVANTI")

            speak("Goodbye.")

            active = False

        else:

            response = get_completion(command)

            speak(response)

