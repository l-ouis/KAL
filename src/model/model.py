import numpy as np
import tensorflow as tf
import sys

class AccompanimentModel(tf.keras.Model):

    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    @tf.function
    def call(self, melody, harmony):
        # melody = tf.keras.layers.Embedding(input_dim=self.decoder.vocab_size, output_dim=self.decoder.hidden_size)(melody)
        # harmony = tf.keras.layers.Embedding(input_dim=self.decoder.vocab_size, output_dim=self.decoder.hidden_size)(harmony)
        output = self.decoder(melody, harmony)
        return output  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        indices = tf.range(len(train_captions))
        shuffled_indices = tf.random.shuffle(indices)
        shuffled_captions = tf.gather(train_captions, shuffled_indices)
        shuffled_image_features = tf.gather(train_image_features, shuffled_indices)

        num_batches = int(len(train_captions) / batch_size)
        total_loss = total_seen = total_correct = 0

        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_image_features = shuffled_image_features[start:end, 1:]
            decoder_input = shuffled_captions[start:end, :-1]
            decoder_labels = shuffled_captions[start:end, 1:]
            padding_index = 0
  
            with tf.GradientTape() as tape:
                probs = self(batch_image_features, decoder_input)
                mask = tf.where(decoder_labels != 0, 1, 0)
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(probs, decoder_labels, mask)
                accuracy = self.accuracy_function(probs, decoder_labels, mask)
                
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        return avg_loss, avg_acc, avg_prp

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):
            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, 1:]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)


def accuracy_function(prbs, labels, mask):
    correct_classes = tf.argmax(tf.cast(prbs, dtype=tf.int64), axis=-1) == tf.cast(labels, dtype=tf.int64)
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):
    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss