import base64
from io import BytesIO

import numpy as np
import soundfile as sf
import speech_recognition as sr
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment


def speech_to_text(audio_bytes):
    """Convert speech audio to text using speech recognition"""
    try:
        # Convert the UploadedFile to bytes
        if hasattr(audio_bytes, "read"):
            try:
                audio_bytes = audio_bytes.read()
            except Exception as e:
                return None

        # Convert the audio bytes to numpy array
        try:
            audio_data, sample_rate = sf.read(BytesIO(audio_bytes))

        except Exception as e:
            return None

        # Create a BytesIO object with the audio in WAV format
        try:
            wav_io = BytesIO()
            wav_io.seek(0)
        except Exception as e:
            return None

        # Use speech recognition to convert to text
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")
            return None
        except sr.UnknownValueError:
            st.error("Speech recognition could not understand audio")
            return None
        except Exception as e:
            st.error(f"Error in speech recognition: {e}")
            return None
    except Exception as e:
        st.error(f"Error converting speech to text: {e}")
        return None


def text_to_speech(text, speed=1.5):
    """Convert text to speech and return audio data with speed adjustment"""
    try:
        tts = gTTS(text=text, lang="en")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        # Use pydub to adjust speed
        sound = AudioSegment.from_mp3(audio_bytes)
        # Change speed by modifying frame_rate
        faster_sound = sound._spawn(
            sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * speed)}
        )
        faster_sound = faster_sound.set_frame_rate(sound.frame_rate)

        # Export to bytes
        output = BytesIO()
        faster_sound.export(output, format="mp3")
        output.seek(0)
        return output.read()
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None


def get_audio_player(audio_bytes):
    """Return HTML audio player with the audio data"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        return f"""
        <audio controls autoplay=false>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    return ""
