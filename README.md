# metronome
simple metronome sound generator.

![metronome_image](https://github.com/user-attachments/assets/ba5b1b77-1cad-4ad2-a3f6-40738eee945e)

## usage

* BPM: Input positive number.
* Time Signature: Input must be a positive integer in the numerator, and the denominator must be one of 2, 4, 8, 16, or 32.
* Duration(sec): Input positive number.
* Sample Rate: Select from 44100, 48000, 88200, 96000.
* Preview Button: You can check the sound.
* Stop Button: You can stop the preview sound.
* Export WAV Button: Export metronome sound to .wav file.

## requirements

If you want to preview sound, install simpleaudio for Mac/Linux.

```
pip install simpleaudio
```

Windows uses `winsound` instead.

