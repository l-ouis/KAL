from music21 import *
from tqdm import tqdm
import numpy as np
import os
from miditok import REMI, TokenizerConfig
from symusic import Score
import pickle
import copy

# What we need to make:
# Dictionary relating each token to a np array
# Strip each token 




# This script differs from valid_midi.py in that we use an alternative dataset instead of the Lakh-based dataset.
# Since this dataset does not have the same two-channel structure, we will do some crude assumptions and split 
# melody and harmony off of of the average pitch of the midi file.

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

config = TokenizerConfig(num_velocities=8)
tokenizer = REMI(config)


raw_data_path = "src/data/maestro_dataset"
valid_data_path = "src/data/valid_midi_maestro"
count = 0
input_tokens_list = []
label_tokens_list = []
break_out = False
for root, dirs, files in os.walk(raw_data_path):
    total_files = len(files)
    for file in files:
        try:
            file_path = os.path.join(root, file)
            score = converter.parse(file_path)
            
            combined_score = stream.Score()
            for part in score.parts:
                combined_score.insert(0, part)

            # Remove all preceding rests
            for element in combined_score.recurse():
                if isinstance(element, note.Note):
                    break
                elif isinstance(element, note.Rest):
                    combined_score.remove(element)
            # Quantize the score
            print("Quantizing score")
            combined_score.quantize(inPlace=True)
            combined_score.write('midi', fp='quantized_score.mid')
            print("Quantized score saved")
            # Calculate the average pitch value of the whole score
            all_pitches = []
            for element in combined_score.pitches:
                all_pitches.append(element.midi)
            if all_pitches:
                average_pitch = sum(all_pitches) / len(all_pitches)
            else:
                average_pitch = 60  # Default to middle C if no pitches are found
            print("Average pitch of the score:", average_pitch)

            
            # lower_part = combined_score.deepcopy()
            # upper_part = combined_score.deepcopy()
            lower_part = copy.deepcopy(combined_score)
            upper_part = copy.deepcopy(combined_score)
            
            
            # Remove all notes/chords in lower_part that are above average_pitch
            lower_remove = []
            print(len(lower_part.recurse()))
            for element in lower_part.recurse():
                try:
                    if element.pitch.midi > average_pitch:
                        lower_part.remove(element, recurse=True)
                        lower_remove.append(element)
                except:
                    pass
                if isinstance(element, note.Note):
                    if element.pitch.midi > average_pitch:
                        lower_part.remove(element, recurse=True)
                        lower_remove.append(element)
                elif isinstance(element, chord.Chord):
                    for chord_note in element.notes: 
                        if chord_note.pitch.midi > average_pitch:
                            lower_part.remove(chord_note, recurse=True)
                            lower_part.remove(element)
                            lower_remove.append(chord_note)
            print(len(lower_remove))
            # lower_part.remove(lower_remove, recurse=True)
            print(len(lower_part.recurse()))

            # Remove all notes/chords in upper_part that are below average_pitch
            upper_remove = []
            print(len(upper_part.recurse()))
            for element in upper_part.recurse():
                try:
                    if element.pitch.midi < average_pitch:
                        upper_part.remove(element, recurse=True)
                        upper_remove.append(element)
                except:
                    pass
                if isinstance(element, note.Note):
                    if element.pitch.midi < average_pitch:
                        upper_part.remove(element, recurse=True)
                        upper_remove.append(element)
                elif isinstance(element, chord.Chord):
                    for chord_note in element.notes:
                        if chord_note.pitch.midi < average_pitch:
                            upper_part.remove(chord_note, recurse=True)
                            upper_part.remove(element)
                            upper_remove.append(chord_note)
            print(len(upper_remove))
            # upper_part.remove(upper_remove, recurse=True)
            print(len(upper_part.recurse()))
            # for element in combined_score.recurse():
            #     if isinstance(element, note.Note):
            #         if element.pitch.midi < average_pitch:
            #             lower_part.append(element)
            #         elif element.pitch.midi >= average_pitch:
            #             upper_part.append(element)
            #         else:
            #             lower_part.append(element)
            #             upper_part.append(element)
            #     elif isinstance(element, chord.Chord):
            #         lower_chord_notes = []
            #         upper_chord_notes = []
            #         for chord_note in element.notes:
            #             if chord_note.pitch.midi < average_pitch:
            #                 lower_chord_notes.append(chord_note)
            #             elif chord_note.pitch.midi >= average_pitch:
            #                 upper_chord_notes.append(chord_note)
            #             else:
            #                 lower_chord_notes.append(chord_note)
            #                 upper_chord_notes.append(chord_note)
            #         if lower_chord_notes:
            #             lower_part.append(chord.Chord(lower_chord_notes))
            #         if upper_chord_notes:
            #             upper_part.append(chord.Chord(upper_chord_notes))
            print("Finished splitting midi off middle c")

            # Get each part (right hand, left hand presumably)
            part1 = upper_part
            part2 = lower_part
            
                
            # Save the split MIDI files
            upper_part.write('midi', fp='upper_midi_path.mid')
            lower_part.write('midi', fp='lower_midi_path.mid')
            print("Saved two midi files")
            part1_score = Score('upper_midi_path.mid')
            part2_score = Score('lower_midi_path.mid')
            input_tokens = tokenizer(part1_score)
            label_tokens = tokenizer(part2_score)
            input_tokens_list.append(input_tokens)
            label_tokens_list.append(label_tokens)
            print("midi file saved: ", file_path)
            # input_midi = tokenizer(input_tokens)
            # label_midi = tokenizer(label_tokens)
            # input_midi.dump_midi("out_input.mid")
            # label_midi.dump_midi("out_label.mid")
            count += 1
            print("At file #", count)
                    
            # Should put these in valid_midi
            # When running the count script only, there are 4653 valid files with 2 
            # Maybe check for midis that also have "piano 1" "piano 2"?
        except Exception as e:
            print("Invalid stuff occured when parsing:", file_path)
            print(e)
            continue
        if count > 0:
            break_out = True
            print("Parsed 1 files.")
            break
    if break_out:
        break
with open('input_tokens_maestro.pkl', 'wb') as f:
    pickle.dump(input_tokens_list, f)
with open('label_tokens_maestro.pkl', 'wb') as f:
    pickle.dump(label_tokens_list, f)
with open('tokenizer_maestro.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Successfully parsed ", count, " files")
