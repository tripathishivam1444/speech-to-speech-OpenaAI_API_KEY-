# https://platform.openai.com/docs/guides/text-to-speech

import sounddevice
from scipy.io.wavfile import write
import numpy as np
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()



####################### Recording #######################



fs = 44100

second = int(input("Enter the Recording Time in seconds: "))
print("Recording.....\n")


record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
sounddevice.wait()

amplification_factor = 18.0
amplified_audio = record_voice * amplification_factor

amplified_audio = np.clip(amplified_audio, -1.0, 1.0)


write("input_speech_voice.wav", fs, amplified_audio)
print("Recording is done please check your folder to listen to the recording\n\n")



################### Transcribing  ###################


client = OpenAI()
audio_file= open("input_speech_voice.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file)

print("Transcripted text ------> " , transcription.text)

################ LLM Calling ################




openai_llm = ChatOpenAI(model = "gpt-4o")

AI_responce = openai_llm.invoke( "you have to give the responce in same Language as you give input. \
                                give in very detailed information in same language this is the input -> " \
                                +  transcription.text)
print("\n\n\n\n AI Responce ---->  ", AI_responce.content)


########################## TTS ###########################


# https://platform.openai.com/docs/guides/text-to-speech
from pathlib import Path
from openai import OpenAI
from  dotenv import load_dotenv  
load_dotenv()

client = OpenAI()
speech_file_path = "AI_respoce_gen.mp3"
response = client.audio.speech.create(
  model="gpt-4o-mini-tts",
  voice="coral", # voice options ["alloy, ash, ballad, coral, echo, fable ,onyx, nova, sage ,shimmer"]
  input= AI_responce.content, #"महाराज छत्रपतीं शिवाजी एक वीर अनु महान शासक थे। जिन्होंने मुगल्या सल्तनत को धूल चटादी ",
#   instructions="Speak in a cheerful and positive tone.",
)
response.stream_to_file(speech_file_path)



############################# Autometically Start Playing ##############################


import playsound

def play_audio(file_path):
    try:
        print("Playing audio...")
        playsound.playsound(file_path)
    except Exception as e:
        print(f"Error while playing audio: {e}")

play_audio(speech_file_path)



