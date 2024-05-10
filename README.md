# KAL-Duet
Brown Deep Learning 2024 Final Project

KAL-Duet is a deep learning model that can generate piano accompaniments given an input melodic sequence. It is a Transformer model that uses the attention mechanism to learn the relationship between melodic notes and the harmonic structure accompaniments.


## Dataset

The model is trained on a combination of the Lakh and Maestro datasets, which provide a diverse range of MIDI files. The Lakh dataset offers a large volume of data, while the Maestro dataset provides high-quality piano recordings. Preprocessing information and directions can be found in the README.md files in src/preprocessing and src/data.

## Training

Run `src/main.py` with arguments to train the model. More information can be found at the top of the `main.py` file.

## Accuracy, Outputs

Read the README.md in src/testing for more information on how to generate outputs. With our training, we were able to achieve an accuracy of around 60% on LAKH and Maestro datasets. We were able to generate valid MIDI outputs, but they were often chaotic or dissonant.

## Credits

This project was an attempt at recreating the first paper below, and gathered inspiration from the rest of the linked papers:

https://mct-master.github.io/machine-learning/2022/05/20/kriswent-generating-piano-accompaniments-using-fourier-transforms.html

http://arxiv.org/pdf/2002.03082

https://www.duo.uio.no/bitstream/handle/10852/95694/1/UiO_Master_Thesis_benjamas.pdf

https://magenta.tensorflow.org/performance-rnn