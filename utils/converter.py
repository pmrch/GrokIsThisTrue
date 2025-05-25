import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def convert_audio(input_file: str, output_file: str = "") -> None:
    """
    Convert audio file to 16kHz mono WAV format suitable for speaker recognition.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output WAV file. If None, will use input filename with '_converted.wav'
    """
    print(f"Loading {input_file}...")
    
    # Load audio file and convert to mono, resample to 16kHz
    audio, sr = librosa.load(input_file, sr=16000, mono=True)
    
    # Normalize audio to float32 between -1 and 1
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:  # Avoid division by zero
        audio = audio / np.max(np.abs(audio))
    
    # Ensure minimum length (0.5 seconds)
    min_samples = int(16000 * 0.5)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_converted.wav")
    
    # Save as WAV
    print(f"Saving to {output_file}...")
    sf.write(output_file, audio, 16000, 'FLOAT')
    print("Conversion complete!")

def process_directory(input_dir: str, output_dir: str = "") -> None:
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save converted files. If None, will use input directory
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_converted.wav")
            try:
                convert_audio(input_path, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert audio files to 16kHz mono WAV format for speaker recognition')
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('--output', '-o', help='Output file or directory (optional)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process input as directory')
    
    args = parser.parse_args()
    
    if args.batch:
        process_directory(args.input, args.output)
    else:
        convert_audio(args.input, args.output)

if __name__ == "__main__":
    main()
