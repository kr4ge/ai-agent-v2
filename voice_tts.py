from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions
import os
import sounddevice as sd
import soundfile as sf

load_dotenv()

client = Client(
    user_id=os.getenv("PLAY_HT_USER_ID"),
    api_key=os.getenv("PLAY_HT_API_KEY"),
)

def play_text_to_speech(text, voice="s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json"):
    options = TTSOptions(voice=voice)
    temp_audio_file = "temp_audio.wav"
    
    with open(temp_audio_file, "wb") as audio_file:
        for chunk in client.tts(text, options, voice_engine='PlayDialog-http'):
            audio_file.write(chunk)
    
    data, samplerate = sf.read(temp_audio_file)
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished
    os.remove(temp_audio_file)