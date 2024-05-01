import numpy as np
from music21 import *
import pickle
from miditok import REMI, TokenizerConfig

# 4000 datapoints for now in these pickled lists
with open('input_tokens.pkl', 'rb') as f:
    input_tokens_list = pickle.load(f)
with open('label_tokens.pkl', 'rb') as f:
    label_tokens_list = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("Length of pickled input and label lists:")
print(len(input_tokens_list))
print(len(label_tokens_list))

final_input_ids = []
final_label_ids = []

# Test loop
for (input, label) in zip(input_tokens_list, label_tokens_list):
    # The actual token data is in the first element
    input_tokens = input[0]
    label_tokens = label[0]
    input_ids = input_tokens.ids
    label_ids = label_tokens.ids

    min_length = min(len(input_ids), len(label_ids))
    batches = min_length // 256
    if min_length > 256:
        # Split the input and label ids into batches of 256 until they can't be split anymore
        input_batches = [input_ids[i*256:(i+1)*256] for i in range(batches)]
        label_batches = [label_ids[i*256:(i+1)*256] for i in range(batches)]

        # Each element in these batches will be a single data point.
        # Append these to the final lists
        final_input_ids.extend(input_batches)
        final_label_ids.extend(label_batches)

# Save final_input_ids and final_label_ids to a pickled file
with open('final_input_ids.pkl', 'wb') as f:
    pickle.dump(final_input_ids, f)
with open('final_label_ids.pkl', 'wb') as f:
    pickle.dump(final_label_ids, f)
