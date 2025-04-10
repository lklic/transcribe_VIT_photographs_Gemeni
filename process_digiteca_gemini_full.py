import sys
import csv
import json
import time
import argparse
import io
import itertools
# import os # Removing this import
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError

# --- Configuration ---
CSV_FILE_PATH = Path("process-digiteca-gemeni/digiteca_images.csv")
IMAGE_ROOT_DIR = Path("/home/ubuntu/digiteca_jpg")
OUTPUT_DIR = Path("process-digiteca-gemeni/transcriptions")
API_KEY_FILE = Path("process-digiteca-gemeni/geminikey.secret")
PROMPT_FILE = Path("process-digiteca-gemeni/prompt.txt")

# Gemini Model Configuration (Based on models_config.py)
GEMINI_MODEL_NAME = 'gemini-2.5-pro-preview-03-25'
GEMINI_INPUT_COST_PER_MILLION = 1.25
GEMINI_OUTPUT_COST_PER_MILLION = 10.0
MAX_RETRIES = 2 # Total 3 attempts (initial + 2 retries)

class ProcessingStats:
    """Tracks statistics for the processing run."""
    def __init__(self):
        self.total_cost = 0.0
        self.total_api_time = 0  # Tracks only the time spent in API calls
        self.processed_count = 0
        self.skipped_count = 0
        self.failed_items = []

    def update_success(self, result: Dict[str, Any]):
        self.total_cost += result.get('cost', 0.0)
        self.total_api_time += result.get('request_time', 0)
        self.processed_count += 1

    def add_failure(self, barcode: str, error: str):
        self.failed_items.append({"barcode": barcode, "error": error})

    def increment_skipped(self):
        self.skipped_count += 1

    def get_summary(self, total_elapsed_time: float) -> Dict[str, Any]:
        return {
            "total_script_runtime": round(total_elapsed_time, 2),
            "total_api_time": round(self.total_api_time, 2),
            "total_estimated_cost": round(self.total_cost, 4),
            "images_processed_successfully": self.processed_count,
            "images_skipped (already processed)": self.skipped_count,
            "images_failed": len(self.failed_items),
            "average_cost_per_processed_image": round(self.total_cost / max(1, self.processed_count), 6),
            "average_api_time_per_processed_image": round(self.total_api_time / max(1, self.processed_count), 2),
            "failed_items_details": self.failed_items
        }

class ImageProcessor:
    """Handles interaction with the Gemini API for image transcription."""
    def __init__(self, api_key: str, prompt: str):
        self.prompt = prompt
        genai.configure(api_key=api_key)
        # Safety settings from original script
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.client = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            safety_settings=safety_settings
        )
        # Specify JSON output format
        self.generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )

    def _load_image_bytes(self, image_path: Path) -> Optional[bytes]:
        """Loads image from path and returns bytes, handles errors."""
        try:
            with Image.open(image_path) as img:
                # Ensure image is in RGB format if it has an alpha channel etc.
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save image to an in-memory bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                return buffer.getvalue()
        except FileNotFoundError:
            print(f"  ERROR: Image file not found: {image_path}")
            return None
        except UnidentifiedImageError:
            print(f"  ERROR: Cannot identify image file (corrupted?): {image_path}")
            return None
        except Exception as e:
            print(f"  ERROR: Failed to load image {image_path}: {e}")
            return None

    def _process_gemini(self, img1_path: Path, img2_path: Path) -> Dict[str, Any]:
        """Loads images and sends them to the Gemini API."""
        img1_bytes = self._load_image_bytes(img1_path)
        if img1_bytes is None:
            raise ValueError(f"Failed to load image 1: {img1_path}")

        img2_bytes = self._load_image_bytes(img2_path)
        if img2_bytes is None:
            raise ValueError(f"Failed to load image 2: {img2_path}")

        img1_part = {"mime_type": "image/jpeg", "data": img1_bytes}
        img2_part = {"mime_type": "image/jpeg", "data": img2_bytes}

        prompt_parts = [
            self.prompt + "\n\nPlease provide the result in a JSON format.",
            img1_part,
            img2_part
        ]

        response = self.client.generate_content(
            prompt_parts,
            generation_config=self.generation_config,
            request_options={'timeout': 300} # 5 minute timeout for potentially long requests
        )
        return response

    def _format_output(self, response, row_data: Dict[str, str], request_time: float) -> Dict[str, Any]:
        """Formats the Gemini API response and calculates cost."""
        # Extract needed data from the row
        barcode = row_data.get("Barcode", "")
        box_barcode = row_data.get("Box_barcode", "")
        item_id = row_data.get("ID", "")
        recto_filename = row_data.get("Filename_1", "")
        verso_filename = row_data.get("Filename_2", "")
        work_id = row_data.get("Work_ID", "")

        # Gemini token counts might be in usage_metadata
        # Using 0 as fallback if not present.
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        # Use total if available, otherwise sum input/output
        total_tokens = getattr(response.usage_metadata, 'total_token_count', input_tokens + output_tokens)
        content = response.text

        input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_MILLION
        total_cost = input_cost + output_cost

        try:
            # Gemini should return valid JSON due to response_mime_type
            annotations = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Gemini response was not valid JSON for barcode {barcode}. Saving raw text. Error: {e}")
            annotations = {"raw_response": content} # Store raw text if JSON parsing fails

        return {
            "id": item_id,
            "barcode": barcode,
            "box_barcode": box_barcode,
            "recto_filename": recto_filename,
            "verso_filename": verso_filename,
            "work_id": work_id,
            "model": GEMINI_MODEL_NAME,
            "annotations": annotations,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "cost": total_cost,
            "status": "OK",
            "request_time": round(request_time, 2)
        }

    def process_image_pair(self, row_data: Dict[str, str], img1_path: Path, img2_path: Path) -> Dict[str, Any]:
        """Processes a pair of images with retry logic."""
        barcode = row_data.get("Barcode", "UNKNOWN") # Use barcode for logging
        last_exception = None
        for attempt in range(MAX_RETRIES + 1):
            start_time = time.time()
            try:
                print(f"  Attempt {attempt + 1}/{MAX_RETRIES + 1} for barcode {barcode}...")
                response = self._process_gemini(img1_path, img2_path)
                elapsed_time = time.time() - start_time
                print(f"  API call successful in {elapsed_time:.2f} seconds.")
                # Pass the full row_data dictionary to format_output
                return self._format_output(response, row_data, elapsed_time)
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"  Attempt {attempt + 1} failed after {elapsed_time:.2f}s: {e}")
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt # Exponential backoff (1, 2, 4 seconds...)
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  Max retries reached for barcode {barcode}.")
                    raise last_exception # Re-raise the last exception after all retries fail

def read_csv_rows(file_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Reads rows from the CSV file, handling potential encoding issues."""
    rows = []
    encodings_to_try = ['utf-8', 'latin-1']
    file_encoding = None

    for encoding in encodings_to_try:
        try:
            with file_path.open('r', newline='', encoding=encoding) as csvfile:
                # Test reading the first line to confirm encoding
                csvfile.readline()
                csvfile.seek(0) # Reset file pointer
                reader = csv.DictReader(csvfile)
                header = reader.fieldnames # Get header
                print(f"Successfully opened CSV with encoding: {encoding}")
                file_encoding = encoding
                break # Stop trying encodings if successful
        except (UnicodeDecodeError, FileNotFoundError):
            continue # Try next encoding or handle FileNotFoundError below

    if not file_encoding:
        print(f"ERROR: Could not open or decode CSV file: {file_path}")
        sys.exit(1)

    # Now read the actual data
    try:
        with file_path.open('r', newline='', encoding=file_encoding) as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                # Basic validation: Check if essential keys exist and are not empty
                if not row.get("Barcode") or not row.get("Box_barcode") or not row.get("Filename_1") or not row.get("Filename_2"):
                   print(f"WARNING: Skipping invalid row (missing essential data): {row}")
                   continue
                rows.append(row)
                count += 1
                if limit is not None and count >= limit:
                    print(f"Reached test limit of {limit} rows.")
                    break
    except Exception as e:
        print(f"ERROR: Failed to read CSV data: {e}")
        sys.exit(1)

    print(f"Read {len(rows)} rows from {file_path}.")
    return rows

def main():
    parser = argparse.ArgumentParser(description="Process image pairs using Gemini API based on a CSV file.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first 2 data rows from the CSV for testing (overrides --range)."
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Specify the row range to process in START-END format (e.g., '1-200'). Inclusive. Defaults to all rows."
    )
    args = parser.parse_args()

    start_row = 1
    end_row = None

    if args.range:
        try:
            parts = args.range.split('-')
            if len(parts) != 2:
                raise ValueError("Range must be in START-END format.")
            start_row = int(parts[0])
            end_row = int(parts[1])
            if start_row < 1:
                raise ValueError("Start row must be 1 or greater.")
            if end_row < start_row:
                raise ValueError("End row cannot be less than start row.")
        except ValueError as e:
            print(f"ERROR: Invalid --range format '{args.range}'. {e}")
            sys.exit(1)

    print("--- Starting Image Processing ---")
    print(f"CSV File: {CSV_FILE_PATH}")
    print(f"Image Root: {IMAGE_ROOT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Model: {GEMINI_MODEL_NAME}")
    if args.test:
        print("Test Mode: Enabled (Processing first 2 rows)")
        start_row = 1
        end_row = 2
    elif args.range:
        print(f"Processing Range: Row {start_row} to {end_row}")
    else:
        print("Processing Range: All rows")


    # --- Setup ---
    start_run_time = time.time()
    stats = ProcessingStats()

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")

    # Read API Key
    try:
        with API_KEY_FILE.open("r") as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("API key file is empty.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Could not read API key from {API_KEY_FILE}. {e}")
        sys.exit(1)

    # Read Prompt
    try:
        with PROMPT_FILE.open("r") as f:
            prompt = f.read()
        if not prompt:
            raise ValueError("Prompt file is empty.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Could not read prompt from {PROMPT_FILE}. {e}")
        sys.exit(1)

    # Initialize Processor
    processor = ImageProcessor(api_key, prompt)

    # --- Read CSV Data & Process Images ---
    # We'll read and process row by row to handle large files and ranges efficiently
    range_desc = f"{start_row} to {end_row}" if end_row is not None else f"{start_row} to End"
    print(f"Reading CSV and processing rows {range_desc}...")

    processed_in_range_count = 0
    encodings_to_try = ['utf-8', 'latin-1']
    file_encoding = None

    # Determine encoding first
    for encoding in encodings_to_try:
        try:
            with CSV_FILE_PATH.open('r', newline='', encoding=encoding) as csvfile:
                csvfile.readline() # Test read
                file_encoding = encoding
                print(f"Successfully determined CSV encoding: {encoding}")
                break
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    if not file_encoding:
        print(f"ERROR: Could not open or decode CSV file: {CSV_FILE_PATH}")
        sys.exit(1)

    # Process rows within the specified range
    try:
        with CSV_FILE_PATH.open('r', newline='', encoding=file_encoding) as csvfile:
            reader = csv.DictReader(csvfile)
            # Use islice to handle the range efficiently (start is 0-based, so adjust)
            start_index = start_row - 1
            # islice takes None for 'until end', so end_index remains end_row
            end_index = end_row

            row_iterator = itertools.islice(reader, start_index, end_index)

            for current_row_index, row in enumerate(row_iterator, start=start_row):
                barcode = row.get("Barcode", "").strip()
                box_barcode = row.get("Box_barcode", "").strip()
                filename_1_tif = row.get("Filename_1", "").strip()
                filename_2_tif = row.get("Filename_2", "").strip()

                print(f"\n[Row {current_row_index}] Processing Barcode: {barcode}")

                # Validate data needed for path construction
                if not all([barcode, box_barcode, filename_1_tif, filename_2_tif]):
                    print("  ERROR: Missing required data in CSV row. Skipping.")
                    stats.add_failure(barcode or f"Row_{current_row_index}", "Missing required CSV data (Barcode, Box_barcode, Filename_1, Filename_2)")
                    continue

                # Construct paths
                filename_1_jpg = filename_1_tif.replace('.tif', '.jpg')
                filename_2_jpg = filename_2_tif.replace('.tif', '.jpg')
                img1_path = IMAGE_ROOT_DIR / box_barcode / filename_1_jpg
                img2_path = IMAGE_ROOT_DIR / box_barcode / filename_2_jpg
                output_json_path = OUTPUT_DIR / f"{barcode}.json"

                print(f"  Image 1 Path: {img1_path}")
                print(f"  Image 2 Path: {img2_path}")
                print(f"  Output Path: {output_json_path}")

                # Check if output already exists
                if output_json_path.exists():
                    print(f"  Output file already exists. Skipping.")
                    stats.increment_skipped()
                    continue

                # Process the image pair
                try:
                    # Pass the whole row dictionary to process_image_pair
                    result = processor.process_image_pair(row, img1_path, img2_path)
                    # Save the result
                    with output_json_path.open('w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"  Successfully processed and saved: {output_json_path.name}")
                    stats.update_success(result)
                    processed_in_range_count += 1
                except Exception as e:
                    print(f"  ERROR: Failed to process barcode {barcode} (Row {current_row_index}) after retries: {e}")
                    stats.add_failure(barcode, str(e))

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during CSV processing: {e}")
        # Optionally add to stats or exit
        stats.add_failure("CSV_Processing", str(e))


    # --- Final Summary ---
    end_run_time = time.time()
    total_elapsed_time = end_run_time - start_run_time

    print(f"\n--- Processing Complete (Processed {processed_in_range_count} rows in the specified range) ---")
    summary = stats.get_summary(total_elapsed_time)
    print(json.dumps(summary, indent=2))

    if summary["failed_items_details"]:
        print("\nNOTE: Some images failed processing. Check the 'failed_items_details' above.")

# Removing generate_index_file function and the call below
if __name__ == "__main__":
    main()
