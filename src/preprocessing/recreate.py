import numpy as np
from music21 import *
import pickle
from miditok import REMI, TokenizerConfig
import tensorflow as tf

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

print("length of vocab:")
print(tokenizer.vocab)
print(len(tokenizer.vocab))

# Test loop
# for i in range(1):
#     input_tokens = input_tokens_list[i]
#     label_tokens = label_tokens_list[i]
#     # print(input_tokens)
#     print(len(input_tokens[0].events))
#     input_events = input_tokens[0].ids
#     print(input_events)
#     # print(len(input_events))
#     # print(len(set(input_events)))
#     print(len(tokenizer.vocab))
#     # print(len(tokenizer.vocab))
    # print(input_tokens[0].events[15])
    # print(tokenizer.vocab[str(input_tokens[0].events[15])])
    # print(len(input_tokens[0].events))
    # print(len(label_tokens[0].events))
    # print(input_tokens[0].events)

    # input_midi = tokenizer.decode([input_events[:256]])
    # input_midi.dump_midi("out_input.mid")   
    # for j in range(20):
    #     print(input_events[j])     
    # print(len(input_tokens[0].ids))
    # print(len(label_tokens[0].ids))
    # input_midi = tokenizer(input_tokens)
    # label_midi = tokenizer(label_tokens)
    # input_midi.dump_midi("out_input.mid")
    # label_midi.dump_midi("out_label.mid")