# Kinect Cymatics Sound Generator

Transform human movement into mesmerizing sound waves that create cymatics patterns on sand plates.

## Overview

This system uses an Xbox Kinect sensor to track a person's position and movement, converting:
- **Distance from sensor** → **Sound frequency** (closer = higher pitch)
- **Movement speed** → **Sound amplitude** (faster = louder/stronger patterns)

The generated sound waves, when played through a speaker attached to a plate with sand, create beautiful cymatics patterns that change dynamically with your movement.

## How It Works

1. **Kinect Tracking**: Tracks your body position in 3D space
2. **Distance Calculation**: Measures how far you are from the sensor (0.5m - 4m range)
3. **Speed Calculation**: Measures how fast you're moving
4. **Sound Synthesis**:
   - Distance maps to frequency (100-1000 Hz)
   - Speed maps to amplitude (volume)
   - Optional modulation for complex patterns
5. **Real-time Audio**: Outputs continuous sound waves to your speaker

## Requirements

### Hardware
- Xbox Kinect sensor (Xbox One Kinect or Kinect v2)
- Kinect adapter for your computer (if needed)
- Speaker/amplifier system
- Metal/plastic plate for cymatics
- Fine sand, salt, or powder

### Software
- Windows (Kinect SDK v2 required) or USB3 adapter for other OS
- Python 3.7+
- Dependencies (see Installation)

## Installation

### 1. Install Kinect SDK
**Windows:**
- Download and install [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)

**Mac/Linux:**
- Use libfreenect2 or other third-party drivers
- May require USB3 adapter

### 2. Install Python Dependencies

```bash
pip install numpy pyaudio pykinect2
```

If you encounter issues with PyAudio on Mac:
```bash
brew install portaudio
pip install pyaudio
```

### 3. Run the Application

```bash
python kinect_cymatics.py
```

## Configuration

Edit `config.json` to customize the behavior:

```json
{
  "min_distance": 0.5,        // Minimum tracking distance (meters)
  "max_distance": 4.0,        // Maximum tracking distance (meters)
  "min_freq": 100.0,          // Minimum frequency (Hz)
  "max_freq": 1000.0,         // Maximum frequency (Hz)
  "max_speed": 2.0,           // Maximum speed for amplitude mapping (m/s)
  "min_amplitude": 0.1,       // Minimum volume (0.0-1.0)
  "max_amplitude": 0.8,       // Maximum volume (0.0-1.0)
  "wave_type": "sine",        // Wave type: sine, square, triangle, sawtooth
  "use_amplitude_modulation": true,   // Speed modulates amplitude
  "use_frequency_modulation": false,  // Speed modulates frequency
  "fm_depth": 50.0            // FM modulation depth (Hz)
}
```

### Recommended Settings for Different Patterns

**Classic Cymatics (Chladni Patterns):**
```json
{
  "min_freq": 100.0,
  "max_freq": 2000.0,
  "wave_type": "sine",
  "use_amplitude_modulation": false
}
```

**Dynamic/Interactive Patterns:**
```json
{
  "min_freq": 200.0,
  "max_freq": 1500.0,
  "wave_type": "sine",
  "use_amplitude_modulation": true
}
```

**Experimental Complex Patterns:**
```json
{
  "min_freq": 150.0,
  "max_freq": 800.0,
  "wave_type": "triangle",
  "use_amplitude_modulation": true,
  "use_frequency_modulation": true,
  "fm_depth": 30.0
}
```

## Usage Tips

### Getting Started
1. Connect your Kinect sensor
2. Run the program
3. Stand in front of the Kinect (1-3 meters away)
4. Move slowly to see frequency changes
5. Move quickly to increase amplitude

### Creating Patterns
- **Move closer** = Higher frequency = More intricate patterns
- **Move faster** = Stronger vibrations = More pronounced patterns
- **Stand still** = Steady tone = Stable pattern formation
- **Slow movements** = Gradual transitions between patterns

### Optimal Cymatics Setup
- Use a thin metal plate (0.5-1mm thick)
- Attach speaker cone directly to center of plate
- Use fine, dry sand or salt
- Start with low volume and increase gradually
- Different frequencies create different pattern geometries

## Frequency Guide for Cymatics

- **100-200 Hz**: Large, simple patterns, low nodes
- **200-500 Hz**: Medium complexity, good for beginners
- **500-1000 Hz**: Complex patterns, more nodes
- **1000-2000 Hz**: Very intricate patterns, requires fine sand

## Troubleshooting

### Kinect Not Detected
- Ensure Kinect is plugged into USB 3.0 port
- Check that Kinect SDK is installed
- Verify Kinect drivers are working

### No Sound Output
- Check audio device selection
- Verify speaker is connected
- Try increasing `max_amplitude` in config

### Tracking Issues
- Ensure good lighting
- Stand within 0.5-4 meter range
- Avoid cluttered background
- Make sure you're facing the Kinect

### Poor Patterns
- Try different frequencies (edit config)
- Check plate is properly mounted
- Use finer sand/powder
- Adjust speaker volume
- Ensure plate is level

## Advanced Customization

### Custom Wave Functions
Edit the `generate_wave()` method in `kinect_cymatics.py` to add custom waveforms:

```python
elif self.wave_type == 'custom':
    # Your custom wave equation
    wave = np.sin(phases) + 0.3 * np.sin(2 * phases)
```

### Multi-Person Tracking
The current implementation tracks one person. To track multiple people, modify the `get_tracked_body()` method to return all tracked bodies and blend their frequencies.

### Recording Patterns
To save audio output for later playback:
```python
import wave

# Add recording functionality to stream callback
```

## Physics Behind Cymatics

Cymatics visualizes sound waves through mechanical vibrations. When a plate vibrates at specific frequencies:
- **Nodes** (still points) form geometric patterns
- **Antinodes** (high vibration) push sand away
- Sand accumulates at nodes, revealing the pattern

Different frequencies create different standing wave patterns based on the plate's resonant modes.

## Safety Notes

- Keep volume at safe levels
- Wear ear protection if testing high volumes
- Ensure Kinect sensor is secure and stable
- Keep liquids away from electronics

## License

MIT License - Feel free to modify and experiment!

## Credits

Created for interactive cymatics installations and experimental sound art.

---

Have fun creating living, breathing patterns with your movements!
