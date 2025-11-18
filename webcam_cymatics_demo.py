"""
Webcam to Cymatics Sound Generator - DEMO VERSION
Tracks a person using webcam face detection and converts their distance and movement
into sound waves for creating cymatics patterns on sand plates.
Uses OpenCV for video and sounddevice for audio.
"""

import numpy as np
import sounddevice as sd
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

        # Audio setup
        self.sample_rate = 44100
        self.block_size = 2048

        # Start audio stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.block_size,
            dtype='float32'
        )
        self.stream.start()

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

        # Current audio parameters
        self.current_frequency = 440.0
        self.current_amplitude = 0.0

        print("=" * 60)
        print("WEBCAM CYMATICS GENERATOR - DEMO")
        print("=" * 60)
        print(f"Distance range: {self.min_distance}m - {self.max_distance}m")
        print(f"Frequency range: {self.min_freq}Hz - {self.max_freq}Hz")
        print(f"Speed range: 0 - {self.max_speed}m/s")
        print(f"Wave type: {self.wave_type}")
        print("=" * 60)
        print("\nCalibrating... Please face the camera at about 2 meters distance")
        print("(This will take ~1 second)\n")

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
        self.min_amplitude = config.get('min_amplitude', 0.05)
        self.max_amplitude = config.get('max_amplitude', 0.3)

        # Wave type
        self.wave_type = config.get('wave_type', 'sine')

        # Modulation
        self.use_amplitude_modulation = config.get('use_amplitude_modulation', True)
        self.use_frequency_modulation = config.get('use_frequency_modulation', False)
        self.fm_depth = config.get('fm_depth', 50.0)

        # Webcam specific
        self.show_video = config.get('show_video', True)
        self.calibration_distance = config.get('calibration_distance', 2.0)

    def get_tracked_person(self, frame):
        """Get the tracked person's position from webcam frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Draw rectangle
            if self.show_video:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.circle(frame, (x + w//2, y + h//2), 8, (0, 255, 0), -1)

            # Calculate normalized center position
            frame_height, frame_width = frame.shape[:2]
            center_x = (x + w/2) / frame_width
            center_y = (y + h/2) / frame_height

            # Face size as distance proxy
            face_size = (w * h) / (frame_width * frame_height)

            return {
                'x': center_x,
                'y': center_y,
                'face_size': face_size
            }, frame

        return None, frame

    def calibrate_baseline(self, position):
        """Calibrate baseline face size"""
        if self.calibration_count < self.calibration_frames:
            self.calibration_sizes.append(position['face_size'])
            self.calibration_count += 1

            if self.calibration_count == self.calibration_frames:
                self.baseline_face_size = np.median(self.calibration_sizes)
                print(f"✓ Calibration complete!")
                print(f"  Baseline face size: {self.baseline_face_size:.6f}\n")
                print("NOW ACTIVE - Move around to create sounds!")
                print("  • Move closer/farther to change FREQUENCY")
                print("  • Move around to change AMPLITUDE")
                print("  • Press 'q' to quit\n")
                print("=" * 60)

            return False
        return True

    def estimate_distance(self, position):
        """Estimate distance from face size"""
        if self.baseline_face_size is None or self.baseline_face_size == 0:
            return 2.0

        if position['face_size'] <= 0:
            return self.max_distance

        estimated_distance = self.calibration_distance * np.sqrt(
            self.baseline_face_size / position['face_size']
        )

        return estimated_distance

    def calculate_speed(self, current_position, current_time):
        """Calculate movement speed"""
        if self.previous_position is None or self.previous_time is None:
            self.previous_position = current_position
            self.previous_time = current_time
            return 0.0

        dx = current_position['x'] - self.previous_position['x']
        dy = current_position['y'] - self.previous_position['y']

        distance_traveled = np.sqrt(dx**2 + dy**2)
        time_elapsed = current_time - self.previous_time

        if time_elapsed > 0:
            speed = (distance_traveled / time_elapsed) * 2.0
        else:
            speed = 0.0

        self.previous_position = current_position
        self.previous_time = current_time

        self.speed_history.append(speed)
        return np.mean(self.speed_history)

    def map_distance_to_frequency(self, distance):
        """Map distance to frequency (closer = higher)"""
        distance = np.clip(distance, self.min_distance, self.max_distance)
        normalized = (self.max_distance - distance) / (self.max_distance - self.min_distance)
        frequency = self.min_freq + normalized * (self.max_freq - self.min_freq)
        return frequency

    def map_speed_to_amplitude(self, speed):
        """Map speed to amplitude"""
        speed = np.clip(speed, 0, self.max_speed)
        normalized = speed / self.max_speed
        amplitude = self.min_amplitude + normalized * (self.max_amplitude - self.min_amplitude)
        return amplitude

    def generate_wave(self, frequency, amplitude, duration_samples):
        """Generate a waveform"""
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

    def run(self):
        """Main loop"""
        print("\nStarting...\n")

        try:
            frame_count = 0

            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Get tracked person
                position, annotated_frame = self.get_tracked_person(frame)

                if position is not None:
                    # Calibrate if needed
                    if not self.calibrate_baseline(position):
                        if self.show_video:
                            progress = int(30 * self.calibration_count / self.calibration_frames)
                            bar = "█" * progress + "░" * (30 - progress)
                            cv2.putText(
                                annotated_frame,
                                f"Calibrating: [{bar}]",
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )
                            cv2.putText(
                                annotated_frame,
                                f"{self.calibration_count}/{self.calibration_frames}",
                                (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )
                            cv2.imshow('Webcam Cymatics', annotated_frame)

                        # Silence during calibration
                        silence = np.zeros(self.block_size, dtype=np.float32)
                        self.stream.write(silence)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    current_time = time.time()

                    # Calculate parameters
                    distance = self.estimate_distance(position)
                    speed = self.calculate_speed(position, current_time)

                    # Map to audio
                    frequency = self.map_distance_to_frequency(distance)
                    amplitude = self.map_speed_to_amplitude(speed)

                    self.current_frequency = frequency
                    self.current_amplitude = amplitude

                    # Generate and play audio
                    audio_chunk = self.generate_wave(frequency, amplitude, self.block_size)
                    self.stream.write(audio_chunk)

                    # Display on video
                    if self.show_video:
                        # Background overlay
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (640, 120), (0, 0, 0), -1)
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

                        # Display info
                        cv2.putText(annotated_frame, f"Distance: {distance:.2f}m", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Speed: {speed:.2f}m/s", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Freq: {frequency:.1f}Hz", (350, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Amp: {amplitude:.2f}", (350, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        # Frequency bar
                        freq_percent = (frequency - self.min_freq) / (self.max_freq - self.min_freq)
                        bar_width = int(620 * freq_percent)
                        cv2.rectangle(annotated_frame, (10, 90), (630, 110), (50, 50, 50), -1)
                        cv2.rectangle(annotated_frame, (10, 90), (10 + bar_width, 110), (0, 255, 255), -1)

                    # Print status every 10 frames
                    if frame_count % 10 == 0:
                        print(f"\rDist: {distance:.2f}m | Speed: {speed:.2f}m/s | "
                              f"Freq: {frequency:6.1f}Hz | Amp: {amplitude:.2f}  ", end='')

                else:
                    # No person - silence
                    silence = np.zeros(self.block_size, dtype=np.float32)
                    self.stream.write(silence)

                    if self.show_video:
                        cv2.putText(annotated_frame, "No face detected - please face the camera",
                                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if frame_count % 10 == 0:
                        print("\rNo face detected...                                ", end='')

                # Show video
                if self.show_video:
                    cv2.imshow('Webcam Cymatics', annotated_frame)

                # Check quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.stream.stop()
        self.stream.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n\n" + "=" * 60)
        print("Cleanup complete - Thank you!")
        print("=" * 60)


if __name__ == "__main__":
    try:
        generator = WebcamCymaticsGenerator()
        generator.run()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
