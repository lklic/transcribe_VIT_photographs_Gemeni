import os
import json
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Path to the transcriptions directory, relative to the script's parent directory
TRANSCRIPTIONS_DIR = SCRIPT_DIR.parent / "transcriptions"
# Output path for the index file, inside the script's directory
INDEX_FILE_PATH = SCRIPT_DIR / "index.json"

def generate_index_file(output_dir: Path, index_path: Path):
    """Scans the output directory for JSON files and creates an index."""
    print(f"Scanning directory: {output_dir}")
    try:
        # Ensure the target directory exists
        if not output_dir.is_dir():
            print(f"ERROR: Transcriptions directory not found: {output_dir.resolve()}")
            return

        # List .json files, excluding index.json itself
        json_files = sorted([
            f for f in os.listdir(output_dir)
            if f.endswith('.json') and f != index_path.name
        ])

        print(f"Found {len(json_files)} transcription files.")

        # Write the sorted list to the index file
        with open(index_path, 'w') as f:
            json.dump(json_files, f, indent=2) # Use indent for readability

        print(f"Successfully generated index file: {index_path.resolve()} with {len(json_files)} entries.")

    except Exception as e:
        print(f"ERROR: Failed to generate index file: {e}")

if __name__ == "__main__":
    generate_index_file(TRANSCRIPTIONS_DIR, INDEX_FILE_PATH)
