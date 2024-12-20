import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from dotenv import load_dotenv
import os
load_dotenv()

def play_text_to_speech(text, language='en', slow=False):
    tts = gTTS(text=text, lang=language, slow=slow)
    temp_audio_file = "temp_audio.wav"
    tts.save(temp_audio_file)
    
    data, samplerate = sf.read(temp_audio_file)
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished
    os.remove(temp_audio_file)
