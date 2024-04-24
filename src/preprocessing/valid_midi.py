from music21 import *
from tqdm import tqdm
import numpy as np
import os

def get_tempo(time, metronome_boundaries):
    for i in metronome_boundaries:
        if time >= i[0] and time < i[1]:
            return i[2]
    return "Invalid time"

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

                    print("Valid 2-channel,", file_path, count)
                    metronome_boundaries = []
                    for mMetronomeBoundary in score.metronomeMarkBoundaries():
                        start = mMetronomeBoundary[0]
                        end = mMetronomeBoundary[1]
                        if start == end:
                            continue
                        metronomeMarkObj = mMetronomeBoundary[2]
                        rounded_bpm = round(metronomeMarkObj.number / 60) * 60
                        if rounded_bpm > 180:
                            rounded_bpm = 180
                        metronome_boundaries.append((start, end, rounded_bpm))
                    
                    # Take the list of metronome boundaries and squash sections with equal rounded bpm
                    squash_metronome_boundaries = []
                    start_time = metronome_boundaries[0][0]
                    current_tempo = metronome_boundaries[0][2]
                    for i in metronome_boundaries[1:]:
                        if i[2] == current_tempo:
                            end_time = i[1]
                            continue
                        else:
                            squash_metronome_boundaries.append((start_time, end_time, current_tempo))
                            start_time = i[0]
                            end_time = i[1]
                            current_tempo = i[2]
                    # Add last one
                    squash_metronome_boundaries.append((start_time, metronome_boundaries[-1][1], current_tempo))

                    # squash_metronome_boundaries is a list of tuples
                    # each tuple is (start_time, end_time, tempo)

                    # Get each part (right hand, left hand presumably)
                    allparts = score.parts
                    allparts = [part for part in allparts if hasattr(part.first(), 'notes')]
                    part1 = allparts[0]
                    part2 = allparts[1]
                    part1_data = []
                    part2_data = []

                    for myPart, data_list in [(part1, part1_data), (part2, part2_data)]:
                        for measure in myPart:
                            notes = measure.notes
                            measure_tempo = get_tempo(measure.offset, squash_metronome_boundaries)
                            measure_duration = measure.barDuration.quarterLength
                            beats_per_second = measure_tempo / 60
                            seconds_per_beat = 1 / beats_per_second # Need 6 of these per beat!
                            tokens_per_beat = seconds_per_beat * 6
                            measure_duration_tokens = measure_duration * tokens_per_beat

                            # Make empty np array placeholder
                            measure_np_array = np.zeros((int(measure_duration_tokens), 5, 3))
                            # 5 notes per token max (for chords), 3 values per note (pitch, length, volume)

                            for n in notes:
                                notes_np_array = np.zeros((5, 3))
                                # Be wary that we may have a chord
                                note_duration = n.duration.quarterLength
                                note_offset = n.offset
                                note_offset_tokens = note_offset * tokens_per_beat
                                note_duration_tokens = note_duration * tokens_per_beat
                                note_volume = n.volume.velocity

                                pitches = [p.midi for p in n.pitches]
                                if len(pitches) > 5:
                                    pitches = pitches[:5]
                                pitches = pitches + [0] * (5 - len(pitches)) # pad for length 5 np array
                                for i in range(len(pitches)):
                                    if pitches[i] != 0: 
                                        notes_np_array[i][0] = pitches[i]
                                        notes_np_array[i][1] = note_duration_tokens
                                        notes_np_array[i][2] = note_volume
                                measure_np_array[int(note_offset_tokens)] = notes_np_array
                            data_list.append(measure_np_array)

                    part1_np_data = np.concatenate(part1_data, axis=0)
                    part2_np_data = np.concatenate(part2_data, axis=0)

                    np.save('src/data/valid_midi/'+str(count)+'_part1_np_data.npy', part1_np_data)
                    np.save('src/data/valid_midi/'+str(count)+'_part2_np_data.npy', part2_np_data)
                    count += 1
                            
                    # Should put these in valid_midi
                    # When running the count script only, there are 4653 valid files with 2 
                    # Maybe check for midis that also have "piano 1" "piano 2"?
            except Exception as e:
                print("Invalid stuff occured when parsing:", file_path)
                print(e)
                continue
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