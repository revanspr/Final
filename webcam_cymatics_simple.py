"""
Webcam to Cymatics Sound Generator (Simple Version)
Tracks a person using webcam face detection and converts their distance and movement
into sound waves for creating cymatics patterns on sand plates.
Uses only OpenCV for maximum compatibility.
"""

import numpy as np
import pyaudio
import cv2
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

        # Initialize face cascade for person detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )

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
        self.speed_history = deque(maxlen=10)

        # Phase accumulator for continuous wave generation
        self.phase = 0.0

        # Baseline face size for distance estimation
        self.baseline_face_size = None
        self.calibration_frames = 30
        self.calibration_count = 0
        self.calibration_sizes = []

        print("Webcam Cymatics Generator initialized (Simple Version)")
        print(f"Distance range: {self.min_distance}m - {self.max_distance}m")
        print(f"Frequency range: {self.min_freq}Hz - {self.max_freq}Hz")
        print(f"Speed range: 0 - {self.max_speed}m/s")
        print("\nCalibrating... Please face the camera at about 2 meters distance")

    def load_config(self, config_file):
        """Load configuration parameters"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file not found, using defaults")
            config = {}

        # Distance to frequency mapping
        self.min_distance = config.get('min_distance', 0.5)
        self.max_distance = config.get('max_distance', 4.0)
        self.min_freq = config.get('min_freq', 100.0)
        self.max_freq = config.get('max_freq', 1000.0)

        # Speed to amplitude mapping
        self.max_speed = config.get('max_speed', 2.0)
        self.min_amplitude = config.get('min_amplitude', 0.1)
        self.max_amplitude = config.get('max_amplitude', 0.8)

        # Wave type
        self.wave_type = config.get('wave_type', 'sine')

        # Speed affects modulation
        self.use_amplitude_modulation = config.get('use_amplitude_modulation', True)
        self.use_frequency_modulation = config.get('use_frequency_modulation', False)
        self.fm_depth = config.get('fm_depth', 50.0)

        # Webcam specific settings
        self.show_video = config.get('show_video', True)
        self.calibration_distance = config.get('calibration_distance', 2.0)

    def get_tracked_person(self, frame):
        """Get the tracked person's position from webcam frame"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (more reliable than full body)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Use the largest face (closest person)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Draw rectangle if video is shown
            if self.show_video:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 255, 0), -1)

            # Calculate center position (normalized coordinates)
            frame_height, frame_width = frame.shape[:2]
            center_x = (x + w/2) / frame_width
            center_y = (y + h/2) / frame_height

            # Face size as proxy for distance
            face_size = (w * h) / (frame_width * frame_height)

            return {
                'x': center_x,
                'y': center_y,
                'face_size': face_size,
                'face_width': w,
                'face_height': h
            }, frame

        return None, frame

    def calibrate_baseline(self, position):
        """Calibrate baseline face size for distance estimation"""
        if self.calibration_count < self.calibration_frames:
            self.calibration_sizes.append(position['face_size'])
            self.calibration_count += 1

            if self.calibration_count == self.calibration_frames:
                # Use median to avoid outliers
                self.baseline_face_size = np.median(self.calibration_sizes)
                print(f"\nCalibration complete! Baseline face size: {self.baseline_face_size:.6f}")
                print("You can now move around to create sounds!")
                print("Move closer/farther to change frequency")
                print("Move left/right/up/down to change amplitude\n")

            return False
        return True

    def estimate_distance(self, position):
        """Estimate distance based on face size (inverse relationship)"""
        if self.baseline_face_size is None or self.baseline_face_size == 0:
            return 2.0

        # Distance is inversely proportional to apparent face size
        # Prevent division by zero
        if position['face_size'] <= 0:
            return self.max_distance

        estimated_distance = self.calibration_distance * np.sqrt(
            self.baseline_face_size / position['face_size']
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
            # Scale speed to approximate m/s
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
        distance = np.clip(distance, self.min_distance, self.max_distance)
        normalized = (self.max_distance - distance) / (self.max_distance - self.min_distance)
        frequency = self.min_freq + normalized * (self.max_freq - self.min_freq)
        return frequency

    def map_speed_to_amplitude(self, speed):
        """Map movement speed to amplitude"""
        speed = np.clip(speed, 0, self.max_speed)
        normalized = speed / self.max_speed
        amplitude = self.min_amplitude + normalized * (self.max_amplitude - self.min_amplitude)
        return amplitude

    def generate_wave(self, frequency, amplitude, duration_samples):
        """Generate a waveform based on parameters"""
        phase_increment = 2.0 * np.pi * frequency / self.sample_rate
        phases = self.phase + phase_increment * np.arange(duration_samples)
        self.phase = phases[-1] % (2.0 * np.pi)

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

        wave = wave * amplitude
        return wave.astype(np.float32)

    def generate_modulated_wave(self, carrier_freq, amplitude, speed, duration_samples):
        """Generate wave with modulation based on speed"""
        t = np.arange(duration_samples) / self.sample_rate
        wave = self.generate_wave(carrier_freq, 1.0, duration_samples)

        if self.use_amplitude_modulation:
            mod_freq = speed * 5.0
            am_wave = 1.0 + 0.5 * np.sin(2.0 * np.pi * mod_freq * t)
            wave = wave * am_wave

        wave = wave * amplitude
        return wave.astype(np.float32)

    def run(self):
        """Main loop: track person and generate audio"""
        print("\nStarting tracking...")
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
                    print(f"\rDist: {distance:.2f}m | Speed: {speed:.2f}m/s | "
                          f"Freq: {frequency:.1f}Hz | Amp: {amplitude:.2f}  ", end='')

                else:
                    # No person tracked - output silence
                    silence = np.zeros(self.chunk_size, dtype=np.float32)
                    self.stream.write(silence.tobytes())

                    if self.show_video:
                        cv2.putText(
                            annotated_frame,
                            "No face detected - please face the camera",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                    print("\rNo face detected - waiting...                              ", end='')

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
        print("\nCleanup complete")


if __name__ == "__main__":
    generator = WebcamCymaticsGenerator()
    generator.run()
