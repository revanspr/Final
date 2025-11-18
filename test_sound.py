"""
Simple sound test to verify audio is working
"""
import numpy as np
import sounddevice as sd
import time

print("=" * 60)
print("SOUND TEST")
print("=" * 60)
print("\nTesting audio output...")
print("You should hear a 440 Hz tone (musical note A) for 2 seconds")
print()

# Generate a simple 440 Hz sine wave (note A)
sample_rate = 44100
duration = 2.0  # seconds
frequency = 440.0

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration))

# Generate sine wave
amplitude = 0.3  # 30% volume
audio = amplitude * np.sin(2 * np.pi * frequency * t)

print("Playing tone...")
print(f"  Frequency: {frequency} Hz")
print(f"  Duration: {duration} seconds")
print(f"  Volume: {amplitude * 100:.0f}%")
print()

# Play the sound
sd.play(audio.astype(np.float32), sample_rate)
sd.wait()  # Wait until sound finishes

print("Sound test complete!")
print()
print("Did you hear the tone?")
print("  - YES: Audio is working! The cymatics program should work too.")
print("  - NO: Check your volume settings and audio device.")
print("=" * 60)
