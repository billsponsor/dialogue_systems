import whisper
import pyaudio
import wave
import tempfile
import os
import time
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import warnings
from scipy.io import wavfile

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Whisper model
model = whisper.load_model("medium") 

# Initialize Resemblyzer encoder
encoder = VoiceEncoder("cpu")

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

    try:
        while True:
            try:
                # Capture longer audio segments
                frames = []
                for _ in range(0, int(RATE / CHUNK * 30)):  # Capture 30-second chunks
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

                # Step 1: Preprocess audio for Resemblyzer
                wav = preprocess_wav(temp_audio_file.name)

                # Step 2: Extract speaker embeddings in 2-second chunks
                embeddings = []
                timestamps = []
                step = 2 * RATE  # 2-second step
                for i in range(0, len(wav), step):
                    chunk = wav[i:i + step]
                    if len(chunk) > 0:
                        embeddings.append(encoder.embed_utterance(chunk))
                        timestamps.append((i / RATE, (i + step) / RATE))

                embeddings = np.array(embeddings)

                # Step 3: Perform clustering on embeddings
                n_speakers = 2  # Adjust based on the expected number of speakers
                clustering = AgglomerativeClustering(n_clusters=n_speakers, metric="cosine", linkage="average")
                speaker_labels = clustering.fit_predict(embeddings)

                # Step 4: Split audio into speaker-specific segments
                speaker_segments = {i: [] for i in range(n_speakers)}
                for idx, (start_time, end_time) in enumerate(timestamps):
                    speaker = speaker_labels[idx]
                    speaker_segments[speaker].append((start_time, end_time))

                # Step 5: Process each speaker's segments
                for speaker, segments in speaker_segments.items():
                    speaker_audio = []
                    for start_time, end_time in segments:
                        start_sample = int(start_time * RATE)
                        end_sample = int(end_time * RATE)
                        speaker_audio.extend(wav[start_sample:end_sample])

                    # Save speaker's audio to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as speaker_audio_file:
                        wavfile.write(speaker_audio_file.name, RATE, np.array(speaker_audio))

                        # Transcribe speaker's audio
                        transcription = model.transcribe(speaker_audio_file.name, fp16=False)["text"]

                        # Write transcription with speaker label
                        f.write(f"Speaker {speaker}: {transcription.strip()}\n")
                        print(f"Speaker {speaker}: {transcription.strip()}")

                        os.remove(speaker_audio_file.name)

                # Delete the original temporary file after processing
                os.remove(temp_audio_file.name)

            except OSError as e:
                if e.errno == -9981:  # Input overflowed
                    print("Warning: Input overflowed. Adjusting...")
                    time.sleep(0.1)
                    continue
                else:
                    raise e

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