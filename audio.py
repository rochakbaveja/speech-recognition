#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the audio file
sampling_freq, signal = wavfile.read('random_sound.wav')

# Display the params
print('\nSignal shape:', signal.shape)
print('Datatype:', signal.dtype)
print('Signal duration:', round(signal.shape[0] / float(sampling_freq), 2), 'seconds')

# Normalize the signal 
signal = signal / np.power(2, 15)

# Extract the first 50 values
signal = signal[:50]

# Construct the time axis in milliseconds
time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)

# Plot the audio signal
plt.plot(time_axis, signal, color='black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the audio file
sampling_freq, signal = wavfile.read('spoken_word.wav')

# Normalize the values
signal = signal / np.power(2, 15) 

# Extract the length of the audio signal
len_signal = len(signal)

# Extract the half length
len_half = np.ceil((len_signal + 1) / 2.0).astype(np.int)

# Apply Fourier transform
freq_signal = np.fft.fft(signal)

# Normalization
freq_signal = abs(freq_signal[0:len_half]) / len_signal

# Take the square
freq_signal **= 2

# Extract the length of the frequency transformed signal
len_fts = len(freq_signal)

# Adjust the signal for even and odd cases
if len_signal % 2:
    freq_signal[1:len_fts] *= 2
else:
    freq_signal[1:len_fts-1] *= 2

# Extract the power value in dB
signal_power = 10 * np.log10(freq_signal)

# Build the X axis
x_axis = np.arange(0, len_half, 1) * (sampling_freq / len_signal) / 1000.0

# Plot the figure
plt.figure()
plt.plot(x_axis, signal_power, color='black')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Signal power (dB)')
plt.show()


# In[24]:


# generating audio signals 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Output file where the audio will be saved 
output_file = 'generated_audio.wav'

# Specify audio parameters
duration = 4  # in seconds
sampling_freq = 44100  # in Hz
tone_freq = 784 
min_val = -4 * np.pi
max_val = 4 * np.pi

# Generate the audio signal
t = np.linspace(min_val, max_val, duration * sampling_freq)
signal = np.sin(2 * np.pi * tone_freq * t)

# Add some noise to the signal
noise = 0.5 * np.random.rand(duration * sampling_freq)
signal += noise

# Scale it to 16-bit integer values
scaling_factor = np.power(2, 15) - 1
signal_normalized = signal / np.max(np.abs(signal))
signal_scaled = np.int16(signal_normalized * scaling_factor)

# Save the audio signal in the output file 
write(output_file, sampling_freq, signal_scaled)

# Extract the first 200 values from the audio signal 
signal = signal[:200]

# Construct the time axis in milliseconds
time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq) 

# Plot the audio signal
plt.plot(time_axis, signal, color='black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Generated audio signal')
plt.show()


# In[25]:


# synthesizing tone to generate music 

import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Synthesize the tone based on the input parameters
# def tone_synthesizer(freq, duration, amplitude=1.0, sampling_freq=44100):
#     # Construct the time axis 
#     time_axis = np.linspace(0, duration, duration * sampling_freq)

#     # Construct the audio signal
#     signal = amplitude * np.sin(2 * np.pi * freq * time_axis)

#     return signal.astype(np.int16) 

def tone_synthesizer(freq, duration, amplitude=1.0, sampling_freq=44100):
    # Construct the time axis 
    time_axis = np.linspace(0, duration, int(duration * sampling_freq))

    # Construct the audio signal
    signal = amplitude * np.sin(2 * np.pi * freq * time_axis)

    return signal.astype(np.int16) 


if __name__=='__main__':
    # Names of output files
    file_tone_single = 'generated_tone_single.wav'
    file_tone_sequence = 'generated_tone_sequence.wav'

    # Source: http://www.phy.mtu.edu/~suits/notefreqs.html
    mapping_file = 'tone_mapping.json'
    
    # Load the tone to frequency map from the mapping file
    with open(mapping_file, 'r') as f:
        tone_map = json.loads(f.read())
        
    # Set input parameters to generate 'F' tone
    tone_name = 'F'
    duration = 3     # seconds
    amplitude = 12000
    sampling_freq = 44100    # Hz

    # Extract the tone frequency
    tone_freq = tone_map[tone_name]

    # Generate the tone using the above parameters
    synthesized_tone = tone_synthesizer(tone_freq, duration, amplitude, sampling_freq)

    # Write the audio signal to the output file
    write(file_tone_single, sampling_freq, synthesized_tone)

    # Define the tone sequence along with corresponding durations in seconds
    tone_sequence = [('G', 0.4), ('D', 0.5), ('F', 0.3), ('C', 0.6), ('A', 0.4)]

    # Construct the audio signal based on the above sequence 
    signal = np.array([])
    for item in tone_sequence:
        # Get the name of the tone 
        tone_name = item[0]

        # Extract the corresponding frequency of the tone
        freq = tone_map[tone_name]

        # Extract the duration
        duration = item[1]

        # Synthesize the tone
        synthesized_tone = tone_synthesizer(freq, duration, amplitude, sampling_freq) 


        # Append the output signal
#         signal = np.append(signal, synthesized_tone, axis=0)
        signal = np.append(signal, synthesized_tone.astype(np.int16), axis=0)

    # Save the audio in the output file
    write(file_tone_sequence, sampling_freq, signal)


# In[30]:


# feature extractor

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
from python_speech_features import mfcc, logfbank

# Read the input audio file
sampling_freq, signal = wavfile.read('random_sound.wav')

# Take the first 10,000 samples for analysis
signal = signal[:10000]

# Extract the MFCC features 
features_mfcc = mfcc(signal, sampling_freq)

# Print the parameters for MFCC
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

# Plot the features
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

# Extract the Filter Bank features
features_fb = logfbank(signal, sampling_freq)

# Print the parameters for Filter Bank 
print('\nFilter bank:\nNumber of windows =', features_fb.shape[0])
print('Length of each feature =', features_fb.shape[1])

# Plot the features
features_fb = features_fb.T
plt.matshow(features_fb)
plt.title('Filter bank')

plt.show()


# In[32]:


# speech recognizer 


# import os 
# import argparse 
# import warnings

# import numpy as np
# from scipy.io import wavfile 

# from hmmlearn import hmm
# from python_speech_features import mfcc

# # Define a function to parse the input arguments
# def build_arg_parser():
#     parser = argparse.ArgumentParser(description='Trains the HMM-based speech \
#             recognition system')
#     parser.add_argument("--input-folder", dest="input_folder", required=True,
#             help="Input folder containing the audio files for training")
#     return parser

# # Define a class to train the HMM 
# class ModelHMM(object):
#     def __init__(self, num_components=4, num_iter=1000):
#         self.n_components = num_components
#         self.n_iter = num_iter

#         self.cov_type = 'diag' 
#         self.model_name = 'GaussianHMM' 

#         self.models = []

#         self.model = hmm.GaussianHMM(n_components=self.n_components, 
#                 covariance_type=self.cov_type, n_iter=self.n_iter)

#     # 'training_data' is a 2D numpy array where each row is 13-dimensional
#     def train(self, training_data):
#         np.seterr(all='ignore')
#         cur_model = self.model.fit(training_data)
#         self.models.append(cur_model)

#     # Run the HMM model for inference on input data
#     def compute_score(self, input_data):
#         return self.model.score(input_data)

# # Define a function to build a model for each word
# def build_models(input_folder):
#     # Initialize the variable to store all the models
#     speech_models = []

#     # Parse the input directory
#     for dirname in os.listdir(input_folder):
#         # Get the name of the subfolder 
#         subfolder = os.path.join(input_folder, dirname)

#         if not os.path.isdir(subfolder): 
#             continue

#         # Extract the label
#         label = subfolder[subfolder.rfind('/') + 1:]

#         # Initialize the variables
#         X = np.array([])

#         # Create a list of files to be used for training
#         # We will leave one file per folder for testing
#         training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]

#         # Iterate through the training files and build the models
#         for filename in training_files: 
#             # Extract the current filepath
#             filepath = os.path.join(subfolder, filename)

#             # Read the audio signal from the input file
#             sampling_freq, signal = wavfile.read(filepath)
            
#             # Extract the MFCC features
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')
#                 features_mfcc = mfcc(signal, sampling_freq)

#             # Append to the variable X
#             if len(X) == 0:
#                 X = features_mfcc
#             else:
#                 X = np.append(X, features_mfcc, axis=0)
            
#         # Create the HMM model
#         model = ModelHMM()

#         # Train the HMM
#         model.train(X)

#         # Save the model for the current word
#         speech_models.append((model, label))

#         # Reset the variable
#         model = None

#     return speech_models

# # Define a function to run tests on input files
# def run_tests(test_files):
#     # Classify input data
#     for test_file in test_files:
#         # Read input file
#         sampling_freq, signal = wavfile.read(test_file)

#         # Extract MFCC features
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             features_mfcc = mfcc(signal, sampling_freq)

#         # Define variables
#         max_score = -float('inf') 
#         output_label = None 

#         # Run the current feature vector through all the HMM
#         # models and pick the one with the highest score
#         for item in speech_models:
#             model, label = item
#             score = model.compute_score(features_mfcc)
#             if score > max_score:
#                 max_score = score
#                 predicted_label = label

#         # Print the predicted output 
#         start_index = test_file.find('/') + 1
#         end_index = test_file.rfind('/')
#         original_label = test_file[start_index:end_index]
#         print('\nOriginal: ', original_label) 
#         print('Predicted:', predicted_label)

# if __name__=='__main__':
#     args = build_arg_parser().parse_args()
#     input_folder = args.input_folder

#     # Build an HMM model for each word
#     speech_models = build_models(input_folder)

#     # Test files -- the 15th file in each subfolder 
#     test_files = []
#     for root, dirs, files in os.walk(input_folder):
#         for filename in (x for x in files if '15' in x):
#             filepath = os.path.join(root, filename)
#             test_files.append(filepath)

#     run_tests(test_files)


# In[ ]:




