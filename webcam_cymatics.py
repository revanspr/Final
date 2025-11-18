"""
Webcam to Cymatics Sound Generator
Tracks a person using webcam and converts their distance and movement speed into sound waves
for creating cymatics patterns on sand plates.
"""

import numpy as np
import pyaudio
import cv2
import mediapipe as mp
import time
from collections import deque
import json


class WebcamCymaticsGenerator:
    def __init__(self, config_file='config.json'):
        """Initialize webcam tracking and audio synthesis"""

        # Load configuration
        self.load_config(config_file)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Set webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize MediaPipe Pose for person tracking
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

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

        # Baseline body size for distance estimation
        self.baseline_body_height = None
        self.calibration_frames = 30
        self.calibration_count = 0

        print("Webcam Cymatics Generator initialized")
        print(f"Distance range: {self.min_distance}m - {self.max_distance}m")
        print(f"Frequency range: {self.min_freq}Hz - {self.max_freq}Hz")
        print(f"Speed range: 0 - {self.max_speed}m/s")
        print("\nCalibrating... Please stand at about 2 meters from the camera")

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

        # Webcam specific settings
        self.show_video = config.get('show_video', True)
        self.calibration_distance = config.get('calibration_distance', 2.0)  # meters

    def get_tracked_person(self, frame):
        """Get the tracked person's position from webcam frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key body points
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

            # Calculate center position (normalized coordinates)
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4

            # Calculate body height (shoulder to hip distance)
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            body_height = abs(hip_y - shoulder_y)

            # Calculate head to hip distance for better distance estimation
            head_to_hip = abs(nose.y - hip_y)

            # Draw pose landmarks if video is shown
            if self.show_video:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

            return {
                'x': center_x,
                'y': center_y,
                'body_height': body_height,
                'head_to_hip': head_to_hip
            }, frame

        return None, frame

    def calibrate_baseline(self, position):
        """Calibrate baseline body size for distance estimation"""
        if self.calibration_count < self.calibration_frames:
            if self.baseline_body_height is None:
                self.baseline_body_height = position['head_to_hip']
            else:
                # Running average
                self.baseline_body_height = (
                    self.baseline_body_height * self.calibration_count +
                    position['head_to_hip']
                ) / (self.calibration_count + 1)

            self.calibration_count += 1

            if self.calibration_count == self.calibration_frames:
                print(f"\nCalibration complete! Baseline height: {self.baseline_body_height:.4f}")
                print("You can now move around to create sounds!\n")

            return False
        return True

    def estimate_distance(self, position):
        """Estimate distance based on body size (inverse relationship)"""
        if self.baseline_body_height is None or self.baseline_body_height == 0:
            return 2.0  # Default distance

        # Distance is inversely proportional to apparent body size
        # At calibration distance, body_height = baseline_body_height
        # When closer, body appears larger; when farther, body appears smaller
        estimated_distance = self.calibration_distance * (
            self.baseline_body_height / position['head_to_hip']
        )

        return estimated_distance

    def calculate_speed(self, current_position, current_time):
        """Calculate movement speed in frame coordinates"""
        if self.previous_position is None or self.previous_time is None:
            self.previous_position = current_position
            self.previous_time = current_time
            return 0.0

        # Calculate 2D distance traveled in normalized coordinates
        dx = current_position['x'] - self.previous_position['x']
        dy = current_position['y'] - self.previous_position['y']

        distance_traveled = np.sqrt(dx**2 + dy**2)
        time_elapsed = current_time - self.previous_time

        if time_elapsed > 0:
            # Scale speed to approximate m/s (normalized coords * estimated scale factor)
            speed = (distance_traveled / time_elapsed) * 2.0
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
            # This is simplified FM

        # Apply final amplitude
        wave = wave * amplitude

        return wave.astype(np.float32)

    def run(self):
        """Main loop: track person and generate audio"""
        print("\nStarting tracking... Move around to create sound!")
        print("Press 'q' to quit\n")

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Get tracked person position
                position, annotated_frame = self.get_tracked_person(frame)

                if position is not None:
                    # Calibrate if needed
                    if not self.calibrate_baseline(position):
                        if self.show_video:
                            cv2.putText(
                                annotated_frame,
                                f"Calibrating... {self.calibration_count}/{self.calibration_frames}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )
                            cv2.imshow('Webcam Cymatics', annotated_frame)

                        # Output silence during calibration
                        silence = np.zeros(self.chunk_size, dtype=np.float32)
                        self.stream.write(silence.tobytes())

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    current_time = time.time()

                    # Calculate distance and speed
                    distance = self.estimate_distance(position)
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

                    # Display info on video
                    if self.show_video:
                        cv2.putText(
                            annotated_frame,
                            f"Dist: {distance:.2f}m | Speed: {speed:.2f}m/s",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            annotated_frame,
                            f"Freq: {frequency:.1f}Hz | Amp: {amplitude:.2f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                    # Print status
                    print(f"\rDistance: {distance:.2f}m | Speed: {speed:.2f}m/s | "
                          f"Freq: {frequency:.1f}Hz | Amp: {amplitude:.2f}", end='')

                else:
                    # No person tracked - output silence
                    silence = np.zeros(self.chunk_size, dtype=np.float32)
                    self.stream.write(silence.tobytes())

                    if self.show_video:
                        cv2.putText(
                            annotated_frame,
                            "No person detected",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                    print("\rNo person detected - waiting...                    ", end='')

                # Show video feed
                if self.show_video:
                    cv2.imshow('Webcam Cymatics', annotated_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("Cleanup complete")


if __name__ == "__main__":
    generator = WebcamCymaticsGenerator()
    generator.run()
