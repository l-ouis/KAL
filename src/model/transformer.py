import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_eager_execution()


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask
        self.positional_embedding = tf.Variable(tf.random.normal((2 * 256 - 1, 170)))

    def call(self, inputs):
        """
        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        # print("I'm here!")
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys
        embedding_dim       = Q.get_shape()[2]
        batch_size          = Q.get_shape()[0]
        # print("I'm here2!")
        # print("input shape:", np.shape(K), np.shape(Q))
        # print("query size n window size", window_size_queries, window_size_keys)

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        # print("I'm here3!")
        # print(np.shape(mask_vals))
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        # print("I'm here4!")
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])
        # print("I'm here5!")

        atten_score = tf.matmul(Q, K, transpose_b=True) 
        # print("atten score, atten mask")
        # print(np.shape(atten_score), np.shape(atten_mask))
        if self.use_mask == True:
            atten_score += atten_mask
        attention_weights = tf.nn.softmax(atten_score / np.sqrt(K.get_shape()[2]), axis=-1) #window_size_keys
        # print("I'm here6!")

        # # calculate relative positional encoding here (source: https://medium.com/@ngiengkianyew/what-is-relative-positional-encoding-7e2fbaa3b510)
        positional_embedding_transposed = tf.transpose(self.positional_embedding)
        relative_positional_encoding = tf.matmul(Q, positional_embedding_transposed)
        padding = tf.zeros((batch_size, window_size_queries, 1))
        rpe_padded = tf.concat([relative_positional_encoding, padding], axis=-1)
        rpe_padded_flat = tf.reshape(rpe_padded, (tf.shape(rpe_padded)[0], -1))
        padding_2 = tf.zeros((batch_size, window_size_queries-1))
        rpe_padded_flat_padded = tf.concat([rpe_padded_flat, padding_2], axis=-1)
        rpe_padded_flat_padded_reshape = tf.reshape(rpe_padded_flat_padded, [batch_size, window_size_queries+1, 2*window_size_queries-1])
        rpe_final = rpe_padded_flat_padded_reshape[:, :-1, -window_size_queries:]
        
        # print("Attention weights shape", attention_weights.shape)
        # print("RPE shape", rpe_final.shape)
        return attention_weights #tf.add(attention_weights, rpe_final)

class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention
        self.K = self.add_weight(name = "k", shape=[input_size, output_size])
        self.Q = self.add_weight(name = "q", shape=[input_size, output_size])
        self.V = self.add_weight(name = "v", shape=[input_size, output_size])
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        # print("I'm here!")
        # print("Shapes:", np.shape(inputs_for_keys), np.shape(inputs_for_values), np.shape(inputs_for_queries))
        K = tf.tensordot(inputs_for_keys, self.K, axes = 1)
        V = tf.tensordot(inputs_for_values, self.V, axes = 1)
        Q = tf.tensordot(inputs_for_queries, self.Q, axes = 1)
        # print("I'm here 2!")

        attn_matrix = self.attn_mtx([K, Q])
        return tf.matmul(attn_matrix, V)


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.use_mask = use_mask
        self.attention_head1 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.attention_head2 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.attention_head3 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.dense_res = tf.keras.layers.Dense(self.emb_sz)
        

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        k1, v1, q1 = inputs_for_keys, inputs_for_values, inputs_for_queries 
        k2, v2, q2 = inputs_for_keys, inputs_for_values, inputs_for_queries  
        k3, v3, q3 = inputs_for_keys, inputs_for_values, inputs_for_queries  
        res1 = self.attention_head1(k1, v1, q1)
        res2 = self.attention_head2(k2, v2, q2)
        res3 = self.attention_head3(k3, v3, q3)
        combined = tf.concat([res1, res2, res3], axis=-1)
        return self.dense_res(combined)
        


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.ff_layer = tf.keras.layers.Dense(emb_sz, activation='relu')

        self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  if not multiheaded else MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) if not multiheaded else MultiHeadedAttention(emb_sz, False)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.layer_norm3 = tf.keras.layers.LayerNormalization()

        self.encoder_atten      = AttentionHead(emb_sz, emb_sz, True)  if not multiheaded else MultiHeadedAttention(emb_sz, True)
        self.encoder_ff_layer = tf.keras.layers.Dense(emb_sz, activation='relu')
        self.encoder_layer_norm1 = tf.keras.layers.LayerNormalization()
        self.encoder_layer_norm2 = tf.keras.layers.LayerNormalization()


    @tf.function
    def call(self, outut_embd, input_embd):
        """
        This functions calls a transformer block.
        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        # print("calling self_atten")
        # print("shapes:", np.shape(inputs), np.shape(context_sequence))
        # print(context_sequence)
        # print(inputs)
        
        # encoder block 
        masked_attn = self.encoder_atten(input_embd, input_embd, input_embd)
        masked_attn = masked_attn + input_embd 
        masked_attn = self.encoder_layer_norm1(masked_attn)
        ff = self.encoder_ff_layer(masked_attn)
        ff = ff + masked_attn
        encoder_output = self.encoder_layer_norm2(ff)

        masked_attn = self.self_atten(outut_embd, outut_embd, outut_embd)
        # print("got masked_attn")
        masked_attn = masked_attn + outut_embd
        # print("got masked_attn")
        masked_attn = self.layer_norm1(masked_attn)
        # print("got context_sequence")

        # unmasked_attn = self.self_context_atten(context_sequence, context_sequence, masked_attn)
        unmasked_attn = self.self_context_atten(masked_attn, encoder_output, encoder_output)
        # print("got unmasked_attn")
        unmasked_attn = unmasked_attn + masked_attn
        unmasked_attn = self.layer_norm2(unmasked_attn)
        # print("got unmasked_attn layer norm 2")
        feed_forward = self.ff_layer(unmasked_attn)
        # print("got feed_forward")
        feed_forward = feed_forward + unmasked_attn
        # print("got feed_forward")
        feed_forward = self.layer_norm3(feed_forward)
        return tf.nn.relu(feed_forward)


def positional_encoding(length, depth):
    min_freq=1e-4
    position = np.arange(length)
    freqs = min_freq**(2*(np.arange(depth)//2)/depth)
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
    return tf.cast(pos_enc, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        embeddings = self.embedding(x)
        # embeddings = embeddings * tf.sqrt(tf.cast(self.embed_size, dtype=tf.float32))
        # embeddings = tf.add(embeddings, self.pos_encoding)
        return embeddings
    