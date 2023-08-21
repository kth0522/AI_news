import whisper
import csv
from pydub import AudioSegment
import os

# Load the model and transcribe the audio
model = whisper.load_model("medium")
result = model.transcribe("./vocals.wav")

# Specify the fields for csv
fields = ['id', 'start', 'end', 'text']

# Name of csv file
filename = "./pilot_study/audio_samples/ai_0001/transcriptions.csv"

# Writing to csv file
with open(filename, 'w') as csvfile:
    # Create a csv writer object
    csvwriter = csv.writer(csvfile)

    # Write the headers
    csvwriter.writerow(fields)

    # Write the data rows
    for segment in result['segments']:
        csvwriter.writerow([segment[field] for field in fields])

# Load the audio file
audio = AudioSegment.from_wav("./pilot_study/audio_samples/ai_0001/vocals.wav")

# Create a directory for the cropped audio files, if it doesn't exist
output_dir = "./pilot_study/audio_samples/ai_0001//cropped_segments"
os.makedirs(output_dir, exist_ok=True)

# Crop and save each segment
for segment in result["segments"]:
    start = int(segment["start"] * 1000)  # convert to milliseconds
    end = int(segment["end"] * 1000)  # convert to milliseconds
    cropped = audio[start:end]
    cropped.export(os.path.join(output_dir, f"{segment['id']}.wav"), format="wav")
