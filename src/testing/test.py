import tensorflow as tf
import numpy as np
import numpy as np
from music21 import *
import pickle
from miditok import REMI, TokenizerConfig

# 4000 datapoints for now in these pickled lists
with open('final_input_ids.pkl', 'rb') as f:
    input_tokens_list = pickle.load(f)
with open('final_label_ids.pkl', 'rb') as f:
    label_tokens_list = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("Length of pickled input and label lists:")
print(len(input_tokens_list))
print(len(label_tokens_list))

print("length of vocab:")
print(len(tokenizer.vocab))
print(tokenizer.vocab)

# Test loop
for i in range(1):
    input_tokens = np.array(input_tokens_list[i])
    label_tokens = np.array(label_tokens_list[i])

# Load the model
model = tf.keras.models.load_model('src/saved_model')

def gen_caption_temperature(model, image_embedding, wordToIds, padID, temp, window_length):
    """
    Function used to generate a caption using an ImageCaptionModel given
    an image embedding. 
    """
    # idsToWords = {id: word for word, id in wordToIds.items()}
    # unk_token = wordToIds['<unk>']
    # caption_so_far = [wordToIds['<start>']]
    unk_token = 0
    caption_so_far = [4]
    while len(caption_so_far) < window_length:
        caption_input = np.array([caption_so_far + ((window_length - len(caption_so_far)) * [padID])])
        logits = model(np.expand_dims(image_embedding, 0), caption_input[:,:-1])
        logits = logits[0][len(caption_so_far) - 1]
        probs = tf.nn.softmax(logits / temp).numpy()
        next_token = unk_token
        attempts = 0
        while (next_token == unk_token or next_token == 4) and attempts < 10:
            next_token = np.random.choice(len(probs), p=probs)
            attempts += 1
        caption_so_far.append(next_token)
    return caption_so_far

temperature = .05

output = gen_caption_temperature(model, input_tokens, tokenizer.vocab, 0, temperature, 256)

# Print the output
print("output")
print(output)
melody = tokenizer.decode([input_tokens])
melody.dump_midi("test_input.mid")   
input_midi = tokenizer.decode([output])
input_midi.dump_midi("test_output.mid")
