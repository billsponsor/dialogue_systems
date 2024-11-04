import whisper
import pyaudio
import wave
import tempfile
import os
import time
from pyAudioAnalysis import audioSegmentation as aS
import warnings

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Initialize Whisper model
model = whisper.load_model("medium")  # You can try 'small' or 'large' based on your needs

# Set up audio recording parameters
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Prepare pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Output file for transcriptions
output_file = "./test_meeting_transcription.txt"

# Open the file once and keep it open until script exits
with open(output_file, "a") as f:
    print("Listening... Press Ctrl+C to stop recording.")
    
    silence_duration = 2  # Duration to wait before considering it silence (in seconds)
    last_transcribe_time = time.time()
    audio_buffer = []
    
    try:
        while True:
            try:
                # Capture longer audio segments
                frames = []
                for _ in range(0, int(RATE / CHUNK * 30)):  # Capture 20-second chunks
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                # Save audio chunk to a temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    wf = wave.open(temp_audio_file.name, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()

                # Step 1: Perform speaker diarization using pyAudioAnalysis
                segments = aS.speaker_diarization(temp_audio_file.name, n_speakers=2)  # Adjust n_speakers as needed
                
                # Step 2: Transcribe audio using Whisper
                transcription = model.transcribe(temp_audio_file.name, fp16=False)["text"]
                
                # Print transcription to console
                print("Transcribed Text:", transcription.strip())

                # Combine transcription with speaker information
                for i, speaker in enumerate(segments):
                    start_time = i * 30  # Each segment represents 30 seconds
                    end_time = start_time + 30

                    f.write(f"Speaker {speaker}: {transcription.strip()} [from {start_time:.2f} to {end_time:.2f}]\n")

                # Delete temporary file after processing
                os.remove(temp_audio_file.name)

            except OSError as e:
                if e.errno == -9981:  # Input overflowed
                    print("Warning: Input overflowed. Adjusting...")
                    time.sleep(0.1)
                    continue
                else:
                    raise e

            # Monitor silence to determine when to pause or stop
            if time.time() - last_transcribe_time > silence_duration:
                f.write("\n")  # Just a separator for readability

            time.sleep(0.1)  # Small pause between chunks

    except KeyboardInterrupt:
        # Handle exit gracefully and close the audio stream
        print("\nStopping transcription...")
    
    finally:
        # Ensure the stream is stopped and closed
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

print(f"Transcription saved to {output_file}")
