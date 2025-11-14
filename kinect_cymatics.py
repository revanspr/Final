"""
Xbox Kinect to Cymatics Sound Generator
Tracks a person and converts their distance and movement speed into sound waves
for creating cymatics patterns on sand plates.
"""

import numpy as np
import pyaudio
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
from collections import deque
import json


class KinectCymaticsGenerator:
    def __init__(self, config_file='config.json'):
        """Initialize Kinect tracking and audio synthesis"""

        # Load configuration
        self.load_config(config_file)

        # Initialize Kinect
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)

        # Initialize audio
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        # Tracking variables
        self.previous_position = None
        self.previous_time = None
        self.speed_history = deque(maxlen=10)  # Smooth speed over last 10 frames

        # Phase accumulator for continuous wave generation
        self.phase = 0.0

        print("Kinect Cymatics Generator initialized")
        print(f"Distance range: {self.min_distance}m - {self.max_distance}m")
        print(f"Frequency range: {self.min_freq}Hz - {self.max_freq}Hz")
        print(f"Speed range: 0 - {self.max_speed}m/s")

    def load_config(self, config_file):
        """Load configuration parameters"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file not found, using defaults")
            config = {}

        # Distance to frequency mapping
        self.min_distance = config.get('min_distance', 0.5)  # meters
        self.max_distance = config.get('max_distance', 4.0)  # meters
        self.min_freq = config.get('min_freq', 100.0)  # Hz (low frequency for cymatics)
        self.max_freq = config.get('max_freq', 1000.0)  # Hz (high frequency for cymatics)

        # Speed to amplitude mapping
        self.max_speed = config.get('max_speed', 2.0)  # m/s
        self.min_amplitude = config.get('min_amplitude', 0.1)
        self.max_amplitude = config.get('max_amplitude', 0.8)

        # Wave type
        self.wave_type = config.get('wave_type', 'sine')  # sine, square, triangle, sawtooth

        # Speed affects modulation
        self.use_amplitude_modulation = config.get('use_amplitude_modulation', True)
        self.use_frequency_modulation = config.get('use_frequency_modulation', False)
        self.fm_depth = config.get('fm_depth', 50.0)  # Hz

    def get_tracked_body(self):
        """Get the closest tracked body from Kinect"""
        if self.kinect.has_new_body_frame():
            bodies = self.kinect.get_last_body_frame()

            if bodies is not None:
                # Find the first tracked body
                for i in range(0, self.kinect.max_body_count):
                    body = bodies.bodies[i]
                    if body.is_tracked:
                        # Get the spine base joint (center of body)
                        joints = body.joints
                        spine_base = joints[PyKinectV2.JointType_SpineBase]

                        if spine_base.TrackingState != PyKinectV2.TrackingState_NotTracked:
                            return spine_base.Position

        return None

    def calculate_distance(self, position):
        """Calculate distance from Kinect to person"""
        # Z coordinate is the distance from the Kinect
        return position.z

    def calculate_speed(self, current_position, current_time):
        """Calculate movement speed"""
        if self.previous_position is None or self.previous_time is None:
            self.previous_position = current_position
            self.previous_time = current_time
            return 0.0

        # Calculate 3D distance traveled
        dx = current_position.x - self.previous_position.x
        dy = current_position.y - self.previous_position.y
        dz = current_position.z - self.previous_position.z

        distance_traveled = np.sqrt(dx**2 + dy**2 + dz**2)
        time_elapsed = current_time - self.previous_time

        if time_elapsed > 0:
            speed = distance_traveled / time_elapsed
        else:
            speed = 0.0

        self.previous_position = current_position
        self.previous_time = current_time

        # Add to history and return smoothed speed
        self.speed_history.append(speed)
        return np.mean(self.speed_history)

    def map_distance_to_frequency(self, distance):
        """Map distance to frequency (closer = higher frequency)"""
        # Clamp distance to range
        distance = np.clip(distance, self.min_distance, self.max_distance)

        # Inverse mapping: closer distance = higher frequency
        # This creates more interesting patterns as you approach
        normalized = (self.max_distance - distance) / (self.max_distance - self.min_distance)
        frequency = self.min_freq + normalized * (self.max_freq - self.min_freq)

        return frequency

    def map_speed_to_amplitude(self, speed):
        """Map movement speed to amplitude"""
        # Clamp speed to range
        speed = np.clip(speed, 0, self.max_speed)

        # Linear mapping
        normalized = speed / self.max_speed
        amplitude = self.min_amplitude + normalized * (self.max_amplitude - self.min_amplitude)

        return amplitude

    def generate_wave(self, frequency, amplitude, duration_samples):
        """Generate a waveform based on parameters"""
        t = np.arange(duration_samples) / self.sample_rate

        # Calculate phase increment
        phase_increment = 2.0 * np.pi * frequency / self.sample_rate

        # Generate phase array with continuous phase
        phases = self.phase + phase_increment * np.arange(duration_samples)
        self.phase = phases[-1] % (2.0 * np.pi)  # Update phase for next chunk

        # Generate wave based on type
        if self.wave_type == 'sine':
            wave = np.sin(phases)
        elif self.wave_type == 'square':
            wave = np.sign(np.sin(phases))
        elif self.wave_type == 'triangle':
            wave = 2.0 * np.arcsin(np.sin(phases)) / np.pi
        elif self.wave_type == 'sawtooth':
            wave = 2.0 * (phases / (2.0 * np.pi) - np.floor(phases / (2.0 * np.pi) + 0.5))
        else:
            wave = np.sin(phases)

        # Apply amplitude
        wave = wave * amplitude

        return wave.astype(np.float32)

    def generate_modulated_wave(self, carrier_freq, amplitude, speed, duration_samples):
        """Generate wave with modulation based on speed"""
        t = np.arange(duration_samples) / self.sample_rate

        # Base carrier wave
        wave = self.generate_wave(carrier_freq, 1.0, duration_samples)

        # Amplitude modulation based on speed
        if self.use_amplitude_modulation:
            mod_freq = speed * 5.0  # Speed affects modulation rate
            am_wave = 1.0 + 0.5 * np.sin(2.0 * np.pi * mod_freq * t)
            wave = wave * am_wave

        # Frequency modulation based on speed
        if self.use_frequency_modulation:
            mod_freq = speed * 3.0
            phase_mod = self.fm_depth * np.sin(2.0 * np.pi * mod_freq * t)
            # This is simplified FM, more complex implementation would modify phase

        # Apply final amplitude
        wave = wave * amplitude

        return wave.astype(np.float32)

    def run(self):
        """Main loop: track person and generate audio"""
        print("\nStarting tracking... Move around to create sound!")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Get tracked body position
                position = self.get_tracked_body()

                if position is not None:
                    current_time = time.time()

                    # Calculate distance and speed
                    distance = self.calculate_distance(position)
                    speed = self.calculate_speed(position, current_time)

                    # Map to audio parameters
                    frequency = self.map_distance_to_frequency(distance)
                    amplitude = self.map_speed_to_amplitude(speed)

                    # Generate audio chunk
                    if self.use_amplitude_modulation or self.use_frequency_modulation:
                        audio_chunk = self.generate_modulated_wave(
                            frequency, amplitude, speed, self.chunk_size
                        )
                    else:
                        audio_chunk = self.generate_wave(
                            frequency, amplitude, self.chunk_size
                        )

                    # Play audio
                    self.stream.write(audio_chunk.tobytes())

                    # Print status
                    print(f"\rDistance: {distance:.2f}m | Speed: {speed:.2f}m/s | "
                          f"Freq: {frequency:.1f}Hz | Amp: {amplitude:.2f}", end='')

                else:
                    # No person tracked - output silence
                    silence = np.zeros(self.chunk_size, dtype=np.float32)
                    self.stream.write(silence.tobytes())
                    print("\rNo person detected - waiting...", end='')

                time.sleep(0.01)  # Small delay to prevent overwhelming the CPU

        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.kinect.close()
        print("Cleanup complete")


if __name__ == "__main__":
    generator = KinectCymaticsGenerator()
    generator.run()
