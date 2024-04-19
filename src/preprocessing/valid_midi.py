from music21 import *
from tqdm import tqdm
import os

raw_data_path = "src/data/raw"
valid_data_path = "src/data/valid_midi"
count = 0
for root, dirs, files in os.walk(raw_data_path):
    total_files = len(files)
    for file in files:
        if file.endswith(".mid"):
            try:
                file_path = os.path.join(root, file)
                score = converter.parse(file_path)
                if len(score.parts) == 2:
                    count += 1
                    print("Valid 2-channel,", file_path, count)
                    # Should put these in valid_midi
                    # When running the count script only, there are 4653 valid files with 2 
                    # Maybe check for midis that also have "piano 1" "piano 2"?
            except:
                print("Invalid MIDI file:", file_path)
                continue
