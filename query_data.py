import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import voice_tts as vtts

DEFAULT_MODEL_SIZE = "base"
DEFAULT_CHUNK_LENGTH = 10
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are James a professional AI Assistant
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer the question based only on the following context:
{context}

---
Conversation history:
{conversation_history}

---

Answer the question based on the above context: {question}
"""

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False
    
def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def main():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    customer_input_transcription = ""
    
    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            conversation_history = []
            # Record audio chunk
            print("_")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("Customer:{}".format(transcription))
                
                # Add customer input to transcript
                customer_input_transcription += "Customer: " + transcription + "\n"
                
                # Process customer input and get response from AI assistant
                output = query_rag(transcription, conversation_history)
                if output:
                    output = output.lstrip()
                    vtts.play_text_to_speech(output)
                    print("AI Assistant:{}".format(output))
    
    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()  

def query_rag(query_text: str, conversation_history: list):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the conversation history.
    formatted_history = "\n".join(conversation_history) if conversation_history else "None so far."

    # Create the prompt.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        conversation_history=formatted_history, context=context_text, question=query_text
    )

    # Generate the response using the model.
    model = OllamaLLM(model="llama3.2:latest")
    response_text = model.invoke(prompt)

    # Append the current interaction to the conversation history.
    conversation_history.append(f"Customer: {query_text}")
    conversation_history.append(f"Assistant: {response_text}")

    # Optionally, include sources in the response.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Sources: {sources}")
    return response_text

if __name__ == "__main__":
    main()
