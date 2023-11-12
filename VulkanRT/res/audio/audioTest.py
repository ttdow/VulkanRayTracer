import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio stream.
#p = pyaudio.PyAudio()
#stream = p.open(format=FORMAT,
#                channels=CHANNELS,       # Number of audio samples per frame
#                rate=RATE,
#                input=True,
#                frames_per_buffer=CHUNK)

# Initialize the plot.
#fig, ax = plt.subplots()
#x = np.arange(0, 2 * CHUNK, 2)
#line, = ax.plot(x, np.random.rand(CHUNK))

# Update function for real-time audio visualization.
#def update_plot(frame):
#    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
#    normalized_data = (data / (32768.0 / 8.0)) + 0.5
#    line.set_ydata(normalized_data)
#    return line,

# Animate the plot.
#ani = animation.FuncAnimation(fig, update_plot, blit=True, cache_frame_data=False)
#plt.show()

# Close the stream.
#stream.stop_stream()
#stream.close()
#p.terminate()

# Load a stereo audio file.
audio_file = wave.open('edge2.wav', 'rb')

# Read the audio frames
frames = audio_file.readframes(-1)
left_channel = np.frombuffer(frames, dtype=np.int16)[::2]  # Extract left channel
#right_channel = np.frombuffer(frames, dtype=np.int16)[1::2]  # Extract right channel

# Create a time array for x-axis (based on audio length)
time = np.arange(0, len(left_channel)) / audio_file.getframerate()

print(len(left_channel))
print(audio_file.getframerate())
print((len(left_channel) / audio_file.getframerate()) / 60.0)

# Plotting stereo waveforms
fig, axs = plt.subplots(1, figsize=(10, 6))

# Plot left channel
axs.plot(time, left_channel, color='blue')
axs.set_title('Left Channel')
axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')

# Plot right channel
#axs[1].plot(time, right_channel, color='red')
#axs[1].set_title('Right Channel')
#axs[1].set_xlabel('Time')
#axs[1].set_ylabel('Amplitude')

#plt.tight_layout()
plt.show()