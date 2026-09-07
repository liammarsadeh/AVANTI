from faster_whisper import WhisperModel
import sounddevice as sd
import re
from LLM import get_completion
from tts import speak
import numpy as np
from easy_ocr import ocr
import cv2
from object_detection import object_detection

SAMPLE_RATE = 16000


# Initialize Whisper

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)



# Record audio of the user

def record_audio(duration):

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    return audio.flatten()



# convert the speach into text using the `Transcribe`

def transcribe_audio(audio):

    """
    Segment is basically each 
    """

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
# so user can have a time when he speak and LLm won't inturrupt him
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

            rms = np.sqrt(np.mean(chunk ** 2)) # calcualte RMS 
            if rms < SILENCE_THRESHOLD: # compare RMS with the silence threshold 
                silence_time += CHUNK_SIZE / SAMPLE_RATE # sum the silence time
            else:
                silence_time = 0 # if rms > silence threshold then 0's the silence time so user can continue giving the command

            if silence_time >= SILENCE_DURATION: # if the user stay silence for >1 end the command and make the LLM respone for the command
                break

    return np.concatenate(audio_chunks)







active = False

camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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

        # if user didn't say anything
        if not command:
            continue

        # to end chat if user want to say hey avanti again it will create new chat and older chat memory is deleted
        if "bye avanti" in command:

            print("BYE AVANTI")

            speak("Goodbye.")

            active = False

        # ocr if user says and of those command it will be activated automatically
        elif (
            "ocr" in command
            or "read this" in command
            or "read the page" in command
            or "read this page" in command
        ):

            print("OCR REQUESTED")


            # Capture fresh Image so user can have a time to adjust text
            
            for _ in range(3):
                ret, frame = camera.read()


            if not ret:

                speak("I couldn't capture the image.")

                continue

            # Run OCR
            print('Running OCR...')

            speak("Okay, let me read it.")

            text = ocr(frame)

            print("OCR:", text)

            # if there's something in the extracted text
            if text:

                speak(text)

            # if it's empty :
            else:

                speak("I couldn't read any text.")

        # Object Detection if user requested one of those commands
        elif (
            "detect objects" in command
            or "detect object" in command
            or "what is around me" in command
            or "what is in front of me" in command
            or "look around" in command
        ):

            speak("OBJECT DETECTION REQUESTED")

            # it will stays open until the user press 'q' in the future we can use voice command to exit it
            object_detection(camera)


        # Send Command to the LLM and then take response from the LLM and make the TTS converte it to voice command :)
        else:
            response = get_completion(command)

            speak(response)

camera.release()
cv2.destroyAllWindows()