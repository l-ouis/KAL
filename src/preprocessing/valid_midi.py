from music21 import *
from tqdm import tqdm
import numpy as np
import os
from miditok import REMI, TokenizerConfig
from symusic import Score
import pickle

# What we need to make:
# Dictionary relating each token to a np array
# Strip each token 

def get_tempo(time, metronome_boundaries):
    for i in metronome_boundaries:
        if time >= i[0] and time < i[1]:
            return i[2]
    return "Invalid time"

config = TokenizerConfig(num_velocities=8)
tokenizer = REMI(config)

raw_data_path = "src/data/raw"
valid_data_path = "src/data/valid_midi"
count = 0
input_tokens_list = []
label_tokens_list = []
for root, dirs, files in os.walk(raw_data_path):
    total_files = len(files)
    for file in files:
        if file.endswith(".mid"):
            try:
                file_path = os.path.join(root, file)
                print(file_path)
                score = converter.parse(file_path)
                print(len(score.parts))
                if len(score.parts) == 2: # Change in actual script
                    print("Here")

                    print("Valid 2-channel,", file_path, count)

                    # Get each part (right hand, left hand presumably)
                    allparts = score.parts
                    allparts = [part for part in allparts if hasattr(part.first(), 'notes')]
                    part1 = allparts[0]
                    part2 = allparts[1]
                    part1_data = []
                    part2_data = []
                    print("Here")
                    
                    part1.write('midi', fp='temp_part1.mid')
                    part2.write('midi', fp='temp_part2.mid')
                    part1_score = Score('temp_part1.mid')
                    part2_score = Score('temp_part2.mid')
                    input_tokens = tokenizer(part1_score)
                    label_tokens = tokenizer(part2_score)
                    input_tokens_list.append(input_tokens)
                    label_tokens_list.append(label_tokens)
                    # input_midi = tokenizer(input_tokens)
                    # label_midi = tokenizer(label_tokens)
                    # input_midi.dump_midi("out_input.mid")
                    # label_midi.dump_midi("out_label.mid")
                    count += 1
                            
                    # Should put these in valid_midi
                    # When running the count script only, there are 4653 valid files with 2 
                    # Maybe check for midis that also have "piano 1" "piano 2"?
            except Exception as e:
                print("Invalid stuff occured when parsing:", file_path)
                print(e)
                continue
    if count > 4000:
            with open('input_tokens.pkl', 'wb') as f:
                pickle.dump(input_tokens_list, f)
            with open('label_tokens.pkl', 'wb') as f:
                pickle.dump(label_tokens_list, f)
            with open('tokenizer.pkl', 'wb') as f:
                pickle.dump(tokenizer, f)
            break
print("Successfully parsed ", count, " files")


# Tokenization of a MIDI file:
# Split up the file by measure
# Do something with time sig changes (either exclude or represent it somehow)
# 60 bpm, 80 bpm, 120 bpm, 160bpm, 155bpm
# round all of them to the nearest multiple of 30 bpm
# 60 -> 60, 80 -> 90, 120 -> 120, 160 -> 150, 155 -> 150
# Split each second into 6 parts: each one of these is a ‘token’
# Each token is a 2D vector of notes
# Each index of this 2d vector is a pitch:
# [key index, length (same unit as a token), volume]
# [0-87, >0, 0-32]
# shape: (5, 3)
# Midi file shape: (2, measures*24, 5, 3) (first channel right hand, second channel left hand (Accompaniment))

     

# f = note.Note
# f.name = 'F'
# f.octave = 5
# f.pitch.pitchClass = 5
# f.midi = 70
# myFDir = f.duration.Duration (in beats)
# That’s a bit better! So an f is about 698hz (if A4 = 440hz), and it is pitch class 5 (where C = 0, C# and Db = 1, etc.).