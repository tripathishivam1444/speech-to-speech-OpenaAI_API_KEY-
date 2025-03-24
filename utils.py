import sounddevice
from scipy.io.wavfile import write
import numpy as np
import time
import os
import threading
import playsound
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
# Load environment variables (API keys)
load_dotenv()

# Global variables
client = OpenAI()
fs = 44100  # Sample rate
recording_thread = None
stop_recording = False
frames = []  # Global list to store audio frames

def record_audio():
    """
    Record audio from microphone until stop_recording is True
    """
    global stop_recording, recording_thread, frames
    stop_recording = False
    frames = []  # Reset frames
    
    # Create a file path for the recording
    audio_file = "input_speech_voice.wav"
    
    def callback(indata, frame_count, time_info, status):
        """This is called for each audio block"""
        if not stop_recording:
            frames.append(indata.copy())
        
    def record_thread_func():
        global stop_recording, frames
        print("Recording started...")
        
        # Start the sounddevice stream
        with sounddevice.InputStream(samplerate=fs, channels=2, callback=callback):
            while not stop_recording:
                time.sleep(0.1)  # Small delay to prevent CPU hogging
        
        # Once stopped, process and save the audio
        if frames:
            # Concatenate all recorded frames
            recording = np.concatenate(frames, axis=0)
            
            # Amplify the audio
            amplification_factor = 18.0
            amplified_audio = recording * amplification_factor
            amplified_audio = np.clip(amplified_audio, -1.0, 1.0)
            
            # Save the audio to a file
            write(audio_file, fs, amplified_audio)
            print("Recording is done. File saved as:", audio_file)
        else:
            print("No audio recorded")
    
    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_thread_func)
    recording_thread.daemon = True  # Make thread a daemon so it exits when main program exits
    recording_thread.start()
    
    return audio_file

def stop_audio_recording():
    """Stop the ongoing recording"""
    global stop_recording, recording_thread
    
    print("Stopping recording...")
    stop_recording = True
    
    # Wait for recording thread to finish
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish
        print("Recording stopped")
    else:
        print("No active recording to stop")

def transcribe_audio(audio_file_path):
    """Transcribe the audio file using OpenAI's Whisper API"""
    try:
        if not os.path.exists(audio_file_path):
            return "Error: Audio file not found"
            
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return f"Error transcribing audio: {str(e)}"

def get_llm_response(transcription_text):
    """Get response from LLM based on transcription"""
    try:
        openai_llm = ChatOpenAI(model="gpt-4o")
        prompt = (
            "You have to give the response in same language as the input. "
            "Give detailed information in the same language. "
            f"This is the input: {transcription_text}"
        )
        
        response = openai_llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return f"Error getting response: {str(e)}"

def generate_speech(text, voice="coral"):
    """Generate speech from text using OpenAI's TTS API"""
    try:
        speech_file_path = "AI_response_gen.mp3"
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        )
        response.stream_to_file(speech_file_path)
        return speech_file_path
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None



def play_audio(file_path):
    """Play the audio file using pydub."""
    if not file_path or not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return
    
    try:
        print(f"Playing audio: {file_path}")
        audio = AudioSegment.from_file(file_path)
        play(audio)
        print("Audio playback completed")
    except Exception as e:
        print(f"Error playing audio: {e}")
