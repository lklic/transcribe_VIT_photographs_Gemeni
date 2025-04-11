import sys
import csv
import json
import time
import argparse
import io
import itertools
import os
import re # Added
from datetime import datetime # Added
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError

# --- Configuration ---
# Paths relative to the script's expected execution directory (process-digiteca-gemeni)
CSV_FILE_PATH = Path("digiteca_images.csv")
IMAGE_ROOT_DIR = Path("/home/ubuntu/digiteca_jpg") # Absolute path, keep as is
OUTPUT_DIR = Path("transcriptions")
API_KEY_FILE = Path("geminikey.secret")
PROMPT_FILE = Path("prompt_instructions.txt") # Instructions only
PROMPT_STRUCTURE_FILE = Path("prompt_output_structure.txt") # Structure definition with comments
PROMPT_VALIDATION_FILE = Path("prompt_validation.json") # Generated schema without comments
LOG_DIR = Path("logs") # Log directory

# Gemini Model Configuration (Based on models_config.py)
GEMINI_MODEL_NAME = 'gemini-2.5-pro-preview-03-25'
GEMINI_INPUT_COST_PER_MILLION = 1.25
GEMINI_OUTPUT_COST_PER_MILLION = 10.0
MAX_API_RETRIES = 2 # Total 3 attempts for API errors
MAX_VALIDATION_RETRIES = 2 # Total 3 attempts including initial try for validation failures

# Custom Exception for Validation Errors
class ValidationError(Exception):
    pass

class ProcessingStats:
    """Tracks statistics for the processing run."""
    def __init__(self):
        self.total_cost = 0.0
        self.total_api_time = 0
        self.processed_count = 0
        self.skipped_count = 0
        self.api_error_count = 0
        self.validation_error_count = 0
        self.failed_items = [] # Stores more detailed failure info

    def update_success(self, result: Dict[str, Any]):
        self.total_cost += result.get('cost', 0.0)
        self.total_api_time += result.get('request_time', 0)
        self.processed_count += 1

    def add_failure(self, barcode: str, failure_type: str, error: str, attempt: Optional[int] = None):
        self.failed_items.append({
            "barcode": barcode,
            "type": failure_type, # e.g., "API", "Validation", "File Load"
            "error": error,
            "attempt": attempt # Track on which attempt it finally failed
        })
        if failure_type == "API":
            self.api_error_count += 1
        elif failure_type == "Validation":
            self.validation_error_count += 1
        # Other types don't increment specific counters but are logged

    def increment_skipped(self):
        self.skipped_count += 1

    def get_summary(self, total_elapsed_time: float) -> Dict[str, Any]:
        total_failed = len(self.failed_items)
        return {
            "total_script_runtime_seconds": round(total_elapsed_time, 2),
            "total_api_time_seconds": round(self.total_api_time, 2),
            "total_estimated_cost_usd": round(self.total_cost, 4),
            "images_processed_successfully": self.processed_count,
            "images_skipped_existing": self.skipped_count,
            "images_failed_total": total_failed,
            "images_failed_api_error": self.api_error_count,
            "images_failed_validation_error": self.validation_error_count,
            "average_cost_per_processed_image": round(self.total_cost / max(1, self.processed_count), 6),
            "average_api_time_per_processed_image": round(self.total_api_time / max(1, self.processed_count), 2),
            "failed_items_details": self.failed_items
        }

# --- Validation Function ---
def validate_structure(data: Any, schema: Any, path: str = "") -> Tuple[bool, Optional[str]]:
    """
    Recursively checks if data contains all keys defined in the schema and basic dict types match.
    Returns (True, None) if valid, or (False, error_message) if invalid.
    """
    if not isinstance(schema, dict):
        return True, None # Base case: schema is not a dict, nothing to validate here

    if not isinstance(data, dict):
        # Data type doesn't match schema expectation (dict)
        return False, f"Type mismatch at '{path or 'root'}': Expected dict, got {type(data).__name__}"

    for key, schema_value in schema.items():
        current_path = f"{path}.{key}" if path else key
        if key not in data:
            # Key defined in schema is missing in data
            return False, f"Missing key at '{current_path}'"

        # Check type and recurse for nested dictionaries
        if isinstance(schema_value, dict):
            data_value = data[key]
            if not isinstance(data_value, dict):
                 # Check if the data value is a dict if the schema expects one
                 return False, f"Type mismatch at '{current_path}': Expected dict, got {type(data_value).__name__}"
            # Recurse
            is_valid, error_msg = validate_structure(data_value, schema_value, current_path)
            if not is_valid:
                # Ensure error_msg is not None before returning, provide default if needed
                return False, error_msg or f"Validation failed within '{current_path}'"

    # If loop completes without returning False, this level is valid
    return True, None
# --- End Validation Function ---


class ImageProcessor:
    """Handles interaction with the Gemini API for image transcription."""
    def __init__(self, api_key: str, prompt_instructions: str, prompt_structure: str):
        # Concatenate instructions and structure for the full prompt
        self.full_prompt = f"{prompt_instructions.strip()}\n\n{prompt_structure.strip()}"

        genai.configure(api_key=api_key)
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
        self.generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )

    def _load_image_bytes(self, image_path: Path) -> Optional[bytes]:
        """Loads image from path and returns bytes, handles errors."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
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

        # Use the concatenated full prompt
        prompt_parts = [
            self.full_prompt + "\n\nPlease provide the result in a JSON format.", # Re-iterate JSON format request
            img1_part,
            img2_part
        ]

        response = self.client.generate_content(
            prompt_parts,
            generation_config=self.generation_config,
            request_options={'timeout': 300}
        )
        return response

    def _format_output(self, response, row_data: Dict[str, str], request_time: float) -> Dict[str, Any]:
        """Formats the Gemini API response and calculates cost. Returns parsed annotations separately."""
        barcode = row_data.get("Barcode", "")
        box_barcode = row_data.get("Box_barcode", "")
        item_id = row_data.get("ID", "")
        recto_filename = row_data.get("Filename_1", "")
        verso_filename = row_data.get("Filename_2", "")
        work_id = row_data.get("Work_ID", "")

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        total_tokens = getattr(response.usage_metadata, 'total_token_count', input_tokens + output_tokens)
        content = response.text

        input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_MILLION
        total_cost = input_cost + output_cost

        try:
            annotations = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Gemini response was not valid JSON for barcode {barcode}. Error: {e}")
            annotations = {"raw_response": content, "parsing_error": str(e)} # Store raw text and error

        result_dict = {
            "id": item_id,
            "barcode": barcode,
            "box_barcode": box_barcode,
            "recto_filename": recto_filename,
            "verso_filename": verso_filename,
            "work_id": work_id,
            "model": GEMINI_MODEL_NAME,
            "annotations": annotations, # Include potentially invalid annotations
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "cost": total_cost,
            "status": "OK", # Initial status, might change after validation
            "request_time": round(request_time, 2)
        }
        return result_dict, annotations # Return both full result and annotations

    def process_image_pair(self, row_data: Dict[str, str], img1_path: Path, img2_path: Path, schema: Dict[str, Any], is_test_mode: bool = False) -> Dict[str, Any]:
        """Processes a pair of images with API retry and validation retry logic."""
        barcode = row_data.get("Barcode", "UNKNOWN")
        last_api_exception = None
        result_dict = None
        annotations = None

        # --- API Call Loop ---
        for api_attempt in range(MAX_API_RETRIES + 1):
            start_time = time.time()
            try:
                print(f"  API Attempt {api_attempt + 1}/{MAX_API_RETRIES + 1} for barcode {barcode}...")
                response = self._process_gemini(img1_path, img2_path)
                elapsed_time = time.time() - start_time
                print(f"  API call successful in {elapsed_time:.2f} seconds.")
                result_dict, annotations = self._format_output(response, row_data, elapsed_time)
                last_api_exception = None # Reset exception on success
                break # Exit API retry loop on success
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"  API Attempt {api_attempt + 1} failed after {elapsed_time:.2f}s: {e}")
                last_api_exception = e
                if api_attempt < MAX_API_RETRIES:
                    wait_time = 2 ** api_attempt
                    print(f"  Retrying API call in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  Max API retries reached for barcode {barcode}.")
                    # If API fails completely, return an error dict
                    return {
                        "id": row_data.get("ID", ""), "barcode": barcode, "status": "API_ERROR",
                        "error": str(last_api_exception), "attempt": api_attempt + 1
                    }

        # --- Validation Loop (only if API call succeeded) ---
        if result_dict and "raw_response" not in annotations: # Check if JSON parsing failed in _format_output
            validation_error_msg = "Initial validation not run" # Default message
            for validation_attempt in range(MAX_VALIDATION_RETRIES + 1):
                print(f"  Validation Attempt {validation_attempt + 1}/{MAX_VALIDATION_RETRIES + 1} for barcode {barcode}...")
                # Validate the structure within the 'annotations' field of the result
                is_valid, validation_error_msg = validate_structure(result_dict.get("annotations", {}), schema)

                if is_valid:
                    print("  Validation successful.")
                    result_dict["status"] = "OK_Validated"
                    result_dict.pop("error", None) # Remove error field on success
                    return result_dict # Return the successful, validated result

                # Validation failed
                fail_reason = validation_error_msg or "Unknown structure issue"
                print(f"  Validation failed. Reason: {fail_reason}") # Ensure message prints
                if is_test_mode:
                    print(f"  ---> [TEST MODE] Validation Error for Barcode {barcode}: {fail_reason}")
                result_dict["status"] = "VALIDATION_FAILED"
                result_dict["error"] = fail_reason # Store specific validation error

                if validation_attempt < MAX_VALIDATION_RETRIES:
                    print("  Retrying API call to attempt correction...")
                    # Retry API call
                    start_time = time.time()
                    try:
                        response = self._process_gemini(img1_path, img2_path)
                        elapsed_time = time.time() - start_time
                        print(f"  Retry API call successful in {elapsed_time:.2f} seconds.")
                        # Overwrite previous results with new ones
                        result_dict, annotations = self._format_output(response, row_data, elapsed_time)
                        # Correct indentation for this block:
                        if "raw_response" in annotations: # Check if JSON parsing failed on retry
                            print("  Retry response JSON parsing failed.")
                            result_dict["status"] = "VALIDATION_RETRY_PARSE_ERROR"
                            result_dict["error"] = annotations.get("parsing_error", "JSON parsing failed on retry")
                            break # Exit validation loop, keep last error status
                    except Exception as e:
                        # Correct indentation for this block:
                        elapsed_time = time.time() - start_time
                        print(f"  Retry API call failed after {elapsed_time:.2f}s: {e}")
                        result_dict["status"] = "VALIDATION_RETRY_API_ERROR"
                        result_dict["error"] = f"API Error during validation retry: {str(e)}" # Store the API error during retry
                        break # Exit validation loop, keep last error status
                else:
                    print(f"  Max validation retries reached for barcode {barcode}.")
                    # Keep status as VALIDATION_FAILED
                    break # Exit validation loop
        elif result_dict:
             # JSON parsing failed initially
             result_dict["status"] = "INITIAL_PARSE_ERROR"

        # Return the result dict, which will have a non-OK status if validation failed or parsing failed
        return result_dict


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Process image pairs using Gemini API based on a CSV file.")
    parser.add_argument(
        "--test", action="store_true",
        help="Process only the first 2 data rows from the CSV for testing (overrides --range)."
    )
    parser.add_argument(
        "--range", type=str, default=None,
        help="Specify the row range to process in START-END format (e.g., '1-200'). Inclusive. Defaults to all rows."
    )
    args = parser.parse_args()

    start_row = 1
    end_row = None

    if args.range:
        # Range parsing logic... (same as before)
        try:
            parts = args.range.split('-')
            if len(parts) != 2: raise ValueError("Range must be in START-END format.")
            start_row = int(parts[0])
            end_row = int(parts[1])
            if start_row < 1: raise ValueError("Start row must be 1 or greater.")
            if end_row < start_row: raise ValueError("End row cannot be less than start row.")
        except ValueError as e:
            print(f"ERROR: Invalid --range format '{args.range}'. {e}")
            sys.exit(1)

    print("--- Starting Image Processing ---")
    # Print config... (same as before)
    print(f"CSV File: {CSV_FILE_PATH}")
    print(f"Image Root: {IMAGE_ROOT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Log Dir: {LOG_DIR}")
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output and log directories exist.")

    # Load Validation Schema
    try:
        print(f"Loading validation schema from: {PROMPT_VALIDATION_FILE}")
        with open(PROMPT_VALIDATION_FILE, 'r') as f:
            schema = json.load(f)
        print("Validation schema loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Validation schema file not found: {PROMPT_VALIDATION_FILE}")
        print("Please ensure the file exists and contains the valid JSON structure.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse validation schema file: {PROMPT_VALIDATION_FILE} - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load schema: {e}")
        sys.exit(1)


    # Read API Key
    try:
        with API_KEY_FILE.open("r") as f: api_key = f.read().strip()
        if not api_key: raise ValueError("API key file is empty.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Could not read API key from {API_KEY_FILE}. {e}")
        sys.exit(1)

    # Read Prompt Parts
    try:
        with PROMPT_FILE.open("r") as f: prompt_instructions = f.read()
        if not prompt_instructions: raise ValueError("Prompt instructions file is empty.")
        with PROMPT_STRUCTURE_FILE.open("r") as f: prompt_structure = f.read()
        if not prompt_structure: raise ValueError("Prompt structure file is empty.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Could not read prompt files. {e}")
        sys.exit(1)

    # Initialize Processor
    processor = ImageProcessor(api_key, prompt_instructions, prompt_structure)

    # --- Read CSV Data & Process Images ---
    range_desc = f"{start_row} to {end_row}" if end_row is not None else f"{start_row} to End"
    print(f"Reading CSV and processing rows {range_desc}...")
    processed_in_range_count = 0
    # CSV reading logic... (same as before)
    encodings_to_try = ['utf-8', 'latin-1']
    file_encoding = None
    for encoding in encodings_to_try:
        try:
            with CSV_FILE_PATH.open('r', newline='', encoding=encoding) as csvfile:
                csvfile.readline() # Test read
                file_encoding = encoding
                print(f"Successfully determined CSV encoding: {encoding}")
                break
        except (UnicodeDecodeError, FileNotFoundError): continue
    if not file_encoding:
        print(f"ERROR: Could not open or decode CSV file: {CSV_FILE_PATH}")
        sys.exit(1)

    # Process rows within the specified range
    try:
        with CSV_FILE_PATH.open('r', newline='', encoding=file_encoding) as csvfile:
            reader = csv.DictReader(csvfile)
            start_index = start_row - 1
            end_index = end_row
            row_iterator = itertools.islice(reader, start_index, end_index)

            for current_row_index, row in enumerate(row_iterator, start=start_row):
                barcode = row.get("Barcode", "").strip()
                box_barcode = row.get("Box_barcode", "").strip()
                filename_1_tif = row.get("Filename_1", "").strip()
                filename_2_tif = row.get("Filename_2", "").strip()

                print(f"\n[Row {current_row_index}] Processing Barcode: {barcode}")

                if not all([barcode, box_barcode, filename_1_tif, filename_2_tif]):
                    print("  ERROR: Missing required data in CSV row. Skipping.")
                    stats.add_failure(barcode or f"Row_{current_row_index}", "CSV_Data_Missing", "Missing required CSV data")
                    continue

                filename_1_jpg = filename_1_tif.replace('.tif', '.jpg')
                filename_2_jpg = filename_2_tif.replace('.tif', '.jpg')
                img1_path = IMAGE_ROOT_DIR / box_barcode / filename_1_jpg
                img2_path = IMAGE_ROOT_DIR / box_barcode / filename_2_jpg
                output_json_path = OUTPUT_DIR / f"{barcode}.json"

                print(f"  Image 1 Path: {img1_path}")
                print(f"  Image 2 Path: {img2_path}")
                print(f"  Output Path: {output_json_path}")

                if output_json_path.exists():
                    print(f"  Output file already exists. Skipping.")
                    stats.increment_skipped()
                    continue

                # Process the image pair (now includes validation and retries)
                try:
                    # Pass the test mode flag
                    result = processor.process_image_pair(row, img1_path, img2_path, schema, is_test_mode=args.test)

                    # Save the result regardless of status (contains error info if failed)
                    with output_json_path.open('w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    # Update stats based on final status
                    if result.get("status") == "OK_Validated":
                        print(f"  Successfully processed and saved: {output_json_path.name}")
                        stats.update_success(result)
                        processed_in_range_count += 1
                    else:
                        # Log failure based on status
                        failure_type = result.get("status", "Unknown_Error")
                        error_message = result.get("error", "Processing failed with status: " + failure_type)
                        attempt_num = result.get("attempt") # Might be None if not API error
                        print(f"  ERROR: Failed processing barcode {barcode} with status: {failure_type}")
                        stats.add_failure(barcode, failure_type, error_message, attempt_num)

                except Exception as e: # Catch unexpected errors during processing call itself
                    print(f"  CRITICAL ERROR during processing call for barcode {barcode}: {e}")
                    stats.add_failure(barcode, "Processing_Exception", str(e))
                    # Optionally save a minimal error JSON
                    error_result = {"id": row.get("ID", ""), "barcode": barcode, "status": "Processing_Exception", "error": str(e)}
                    with output_json_path.open('w') as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False)


    except Exception as e:
        print(f"ERROR: An unexpected error occurred during CSV processing: {e}")
        stats.add_failure("CSV_Processing", "File_Read_Error", str(e))

    # --- Final Summary & Logging ---
    end_run_time = time.time()
    total_elapsed_time = end_run_time - start_run_time

    print(f"\n--- Processing Complete (Attempted {processed_in_range_count + stats.api_error_count + stats.validation_error_count} rows in the specified range) ---")
    summary = stats.get_summary(total_elapsed_time)

    # Print summary to console
    print("\n--- Run Summary ---")
    print(json.dumps(summary, indent=2))

    # Save summary to timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOG_DIR / f"log_{timestamp}.json"
    try:
        with open(log_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to log file: {log_filename}")
    except Exception as e:
        print(f"\nERROR: Failed to write log file: {e}")

    if summary["images_failed_total"] > 0:
        print("\nNOTE: Some images failed processing. Check the 'failed_items_details' in the log file or console output.")

if __name__ == "__main__":
    main()
