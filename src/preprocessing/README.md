# Preprocessing Data

First, read the README inside src/data to download and extract the proper files.

Then, you have two choices of dataset:

    - Lakh dataset: slightly less clean online MIDI file database with a large amount of shorter files.

        This dataset isn't guaranteed to have piano channels and instead any MIDI file with two channels
        is grabbed.

        Usage: lakh_valid_midi.py -> lakh_length_normalize.py

    - Maestro dataset: a more clean dataset with only piano channels.

        This dataset is guaranteed to be piano and is the one recommended for training.

        However, each MIDI file is only one channel, so the channel is split off Middle C.
        
        Usage: mae_midi_quantize.py -> mae_quantized_normalizer.py

Please run python files from root directory.
