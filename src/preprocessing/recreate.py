import numpy as np
from music21 import *

# 4312 valids for now
for double_midi_channel in range(5):
    file_1 = np.load('src/data/valid_midi/'+str(double_midi_channel)+'_part1_np_data.npy')
    file_2 = np.load('src/data/valid_midi/'+str(double_midi_channel)+'_part2_np_data.npy')


    # Each token is a 6th of a second
    channel_1 = stream.Stream()

    offset = 0 # in 3 * quarter note offset
    for i in file_1:
        valid_notes = i
        # for potential_note in i:
        #     if potential_note[0] != 0:
        #         valid_notes.append(potential_note)
        this_chord = chord.Chord()
        for valid_note in valid_notes:
            #duration_in_120bpm = valid_note[1] * (1/3)
            duration_in_4ths = valid_note[1] / 4
            this_note = note.Note(int(valid_note[0]), quarterLength=duration_in_4ths)
            this_note.volume.velocity = valid_note[2]
            this_chord.add(this_note)
        channel_1.insert(offset / 3, this_chord)
        offset += 1


    channel_2 = stream.Stream()
    offset = 0 # in 3 * quarter note offset
    for i in file_2:
        valid_notes = i
        # for potential_note in i:
        #     if potential_note[0] != 0:
        #         valid_notes.append(potential_note)
        this_chord = chord.Chord()
        for valid_note in valid_notes:
            #duration_in_120bpm = valid_note[1] * (1/3)
            duration_in_4ths = valid_note[1] / 4
            this_note = note.Note(int(valid_note[0]), quarterLength=duration_in_4ths)
            this_note.volume.velocity = valid_note[2]
            this_chord.add(this_note)
        channel_2.insert(offset / 3, this_chord)
        offset += 1

    new_midi = stream.Score()
    new_midi.insert(0, meter.TimeSignature('4/4'))
    new_midi.insert(0, tempo.MetronomeMark(number=120))
    new_midi.insert(0, channel_1)
    new_midi.insert(0, channel_2)

    new_midi.write('midi', fp=str(double_midi_channel)+'.mid')
