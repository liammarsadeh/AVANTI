from LLM import get_completion
from tts import speak


response = get_completion("Where Should i Go?")

speak(response)
