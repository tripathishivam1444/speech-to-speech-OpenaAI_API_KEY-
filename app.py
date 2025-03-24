import streamlit as st
import time
import os
from utils import record_audio, stop_audio_recording, transcribe_audio, get_llm_response, generate_speech, play_audio

# Page configuration
st.set_page_config(page_title="Voice Chat Assistant", layout="wide")
st.markdown("<h1 style='text-align: center; color: #7E57C2;'>Voice Chat Assistant</h1>", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "recording" not in st.session_state:
    st.session_state.recording = False
    
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# Voice selection
voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "ash", "ballad", "sage"]
selected_voice = st.sidebar.selectbox("Select AI Voice", voice_options, index=6)  # Default to 'coral'

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Create separate buttons for start and stop recording at the bottom
col1, col2, col3 = st.columns([1, 1, 1])

# Start recording button
with col1:
    if st.button("🎤 Start Recording", 
                key="start_record_button",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.recording):
        st.session_state.recording = True
        # Start recording in the background
        st.session_state.audio_file = record_audio()
        st.rerun()  # Using st.rerun() instead of experimental_rerun

# Stop recording button
with col3:
    if st.button("⏹️ Stop Recording", 
                key="stop_record_button",
                use_container_width=True,
                type="secondary",
                disabled=not st.session_state.recording):
        if st.session_state.recording:
            # Stop the recording
            stop_audio_recording()
            st.session_state.recording = False
            st.rerun()  # Using st.rerun() instead of experimental_rerun

# Process the recorded audio if available
if st.session_state.audio_file and os.path.exists(st.session_state.audio_file) and not st.session_state.recording:
    with st.spinner("Transcribing..."):
        # Transcribe the audio
        transcription = transcribe_audio(st.session_state.audio_file)
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": transcription})
        
        # Get LLM response
        with st.spinner("Generating response..."):
            llm_response = get_llm_response(transcription)
            
            # Generate speech from LLM response
            speech_file = generate_speech(llm_response, selected_voice)
            
            # Auto-play the audio
            play_audio(speech_file)
            
            # Add AI response to chat
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
        
        # Reset the audio file state after processing
        st.session_state.audio_file = None
        st.rerun()  # Using st.rerun() instead of experimental_rerun