# Webcam Cymatics Generator - User Guide

## What This Does

This project converts your **movement and position** in front of a webcam into **sound frequencies and amplitudes** that can be used to create cymatics patterns on a sand plate or speaker.

## Files Included

### Main Programs

- **`kinect_cymatics.py`** - Original version using Xbox Kinect sensor
- **`webcam_cymatics_demo.py`** - **NEW!** Webcam version (no Kinect needed)
- **`config.json`** - Configuration file for sound parameters

### Test & Utility Files

- **`quick_test.py`** - Quick diagnostic to check if your webcam works
- **`test_webcam.py`** - More detailed webcam test with face detection
- **`run_camera_test.bat`** - Double-click to run the quick webcam test
- **`run_cymatics.bat`** - **Double-click to run the cymatics generator**

### Other Files

- **`requirements.txt`** - Python packages needed
- **`README.md`** - Original project documentation

## How to Use

### Step 1: Test Your Webcam

**Double-click:** `run_camera_test.bat`

This will check if your webcam is accessible. You should see:
- "SUCCESS: Webcam is accessible!"
- The resolution of your camera

If you see errors, check:
- Is your webcam plugged in?
- Is another program using the webcam? (Close Zoom, Teams, etc.)
- Camera permissions in Windows Settings → Privacy → Camera

### Step 2: Run the Cymatics Generator

**Double-click:** `run_cymatics.bat`

A window will appear showing your webcam feed.

#### During Calibration (~1 second):
- **Sit about 2 meters (6-7 feet)** from your webcam
- **Face the camera directly**
- **Stay still** while it calibrates
- You'll see a progress bar: `[████████████░░░░░░]`

#### After Calibration:
The system is now active! You'll hear sounds based on your movement:

**Distance Controls Frequency (Pitch):**
- **Move CLOSER** → Higher frequency (higher pitch, up to 1000 Hz)
- **Move FARTHER** → Lower frequency (lower pitch, down to 100 Hz)

**Movement Speed Controls Amplitude (Volume):**
- **Move FASTER** (left/right/up/down) → Louder sound
- **Stay STILL** → Quieter sound
- **No detection** → Silence

#### On-Screen Display:

You'll see:
- Green rectangle around your detected face
- **Distance** - Estimated distance in meters
- **Speed** - Movement speed in m/s
- **Freq** - Current sound frequency in Hz
- **Amp** - Current amplitude (volume)
- Yellow frequency bar showing pitch visually

#### To Exit:
Press the **'q'** key while the video window is active

## How It Works

### Detection Method
- Uses **OpenCV Haar Cascade** face detection
- Tracks your face position and size
- Estimates distance based on face size (larger face = closer)
- Calculates movement speed from position changes

### Sound Generation
- **Distance** → Mapped to **Frequency** (closer = higher pitch)
- **Speed** → Mapped to **Amplitude** (faster = louder)
- Generates continuous sine waves (or other waveforms from config)
- Outputs through your default audio device

### Cymatics Application
Connect your computer's audio output to:
- A speaker with a sand/salt plate on top
- A subwoofer facing upward with particles
- A tone generator system

The changing frequencies will create different cymatics patterns as you move!

## Configuration

Edit `config.json` to customize:

```json
{
  "min_distance": 0.5,        // Minimum distance in meters
  "max_distance": 4.0,        // Maximum distance in meters
  "min_freq": 100.0,          // Minimum frequency in Hz
  "max_freq": 1000.0,         // Maximum frequency in Hz
  "max_speed": 2.0,           // Maximum speed in m/s
  "min_amplitude": 0.05,      // Minimum volume
  "max_amplitude": 0.3,       // Maximum volume
  "wave_type": "sine",        // Wave type: sine, square, triangle, sawtooth
  "use_amplitude_modulation": true,   // Enable AM based on speed
  "use_frequency_modulation": false,  // Enable FM based on speed
  "fm_depth": 50.0,           // FM depth in Hz
  "show_video": true,         // Show video window
  "calibration_distance": 2.0 // Distance to stand during calibration
}
```

## Troubleshooting

### No Window Appears
- Check if the window is minimized or behind other windows
- Look in your taskbar for "Webcam Cymatics"
- Try running from Command Prompt to see error messages

### "No face detected"
- **Improve lighting** - face the camera with good light on your face
- **Face the camera directly** - profile/side views don't work well
- **Remove obstructions** - hats, masks, hands covering face
- **Distance** - try moving closer or farther
- **Camera angle** - make sure camera is pointing at you

### No Sound
- Check your **volume** is turned up
- Check **default audio device** in Windows sound settings
- The amplitude might be too low - move around more
- If no face is detected, no sound is generated

### Webcam Not Accessible
- Close other programs using the camera (Zoom, Teams, Skype, etc.)
- Check **Windows Settings** → **Privacy & Security** → **Camera**
  - Enable "Let apps access your camera"
  - Enable "Let desktop apps access your camera"
- Try unplugging and replugging external webcams
- Restart your computer

### Program Crashes
- Make sure you installed all dependencies: `pip install opencv-python sounddevice numpy`
- Check if your Python version is compatible (3.7+)
- Look at the error message for specific issues

## Technical Details

### Dependencies
- **Python 3.7+**
- **opencv-python** - Computer vision and webcam access
- **sounddevice** - Audio output
- **numpy** - Numerical computations
- **CFFI** - Required by sounddevice

### System Requirements
- Webcam (built-in or USB)
- Windows/Mac/Linux
- Audio output device
- ~50 MB disk space

### Performance
- ~30 FPS video processing
- 44.1 kHz audio output
- Low latency (<100ms)

## Tips for Best Results

1. **Lighting** - Good, even lighting on your face is crucial
2. **Background** - Plain backgrounds work better for detection
3. **Position** - Center yourself in the camera frame
4. **Calibration** - Stand at exactly 2 meters during calibration for accurate distance estimates
5. **Headphones** - Use headphones to hear the sound clearly without feedback
6. **Speakers** - For cymatics, use a good quality speaker capable of the frequency range

## Creating Cymatics Patterns

To use this for actual cymatics:

1. **Connect audio output** to a speaker/subwoofer
2. **Place speaker facing up**
3. **Put a plate or membrane** on top of the speaker
4. **Add fine particles** (sand, salt, water, powder)
5. **Run the program** and move around
6. **Watch patterns form** as frequencies change

Different frequencies create different geometric patterns!

## Credits

Based on the original Kinect Cymatics project, adapted to use standard webcams for accessibility.

## Support

If you encounter issues:
1. Check this documentation
2. Run `quick_test.py` to diagnose webcam issues
3. Check the error messages in the terminal
4. Verify all dependencies are installed

---

**Enjoy creating sound and cymatics patterns with your movement!**
