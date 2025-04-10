# Digiteca Image Transcription with Gemini

This project contains a Python script (`process_digiteca_gemini_full.py`) designed to process pairs of historical artwork photographs using the Google Gemini API for transcription. It reads image information from a CSV file, fetches corresponding local image files, sends them to the Gemini API with a specific prompt, and saves the structured JSON transcription results.

## Prerequisites

*   Python 3.6+
*   Required Python libraries: `google-generativeai`, `Pillow`

## Setup

1.  **Clone/Download:** Place the project files in your desired directory.
2.  **Install Dependencies:**
    ```bash
    pip install google-generativeai Pillow
    ```
    *(Note: The script uses `pip`, ensure it's available in your environment).*
3.  **API Key:**
    *   Obtain a Google Gemini API key.
    *   Save your API key in a file named `geminikey.secret` within the `process-digiteca-gemeni` directory. The script expects this file to contain only the API key string.
4.  **Image Files:**
    *   Ensure the image files referenced in the CSV are accessible locally. The script expects a root directory structure like `/home/ubuntu/digiteca_jpg/{Box_barcode}/{Filename.jpg}`. Modify the `IMAGE_ROOT_DIR` variable in the script if your image root path is different.
    *   The script assumes the images are in `.jpg` format. It will automatically replace `.tif` extensions found in the CSV filenames with `.jpg` when constructing file paths.
5.  **CSV File:**
    *   Place the `digiteca_images.csv` file in the `process-digiteca-gemeni` directory. It should contain columns like `Barcode`, `Box_barcode`, `Filename_1`, `Filename_2`.
6.  **Prompt File:**
    *   The `prompt.txt` file contains the instructions sent to the Gemini API along with the images. Ensure this file exists in the `process-digiteca-gemeni` directory.

## File Structure Overview

```
process-digiteca-gemeni/
├── process_digiteca_gemini_full.py  # The main processing script
├── digiteca_images.csv          # Input CSV with image metadata
├── geminikey.secret             # File containing your Gemini API key (!!! DO NOT COMMIT !!!)
├── prompt.txt                   # Prompt used for the Gemini API call
├── models_config.py             # (Used for reference, costs hardcoded in main script)
├── process_images.py            # (Original benchmarking script, not used by full processing)
├── transcriptions/              # Output directory for JSON results (created by script)
│   └── ...
├── digiteca-viewer/             # Simple web viewer files
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── generate_index.py        # Script to create index for the viewer
│   └── index.json               # Generated index of transcription files (used by viewer)
└── README.md                    # This file
```

*(Assumes image files are located separately, e.g., under `/home/ubuntu/digiteca_jpg/`)*

## Workflow

1.  **Process Transcriptions:** Run the main processing script to generate/update the JSON transcription files in the `transcriptions/` directory.
    ```bash
    python process-digiteca-gemeni/process_digiteca_gemini_full.py [OPTIONS]
    ```
    **Processing Options:**

*   **No arguments:** Processes all rows in the `digiteca_images.csv` file.
    ```bash
    python process-digiteca-gemeni/process_digiteca_gemini_full.py
    ```
*   `--range START-END`: Processes a specific range of rows (1-based index, inclusive).
    ```bash
    # Process rows 1 through 200
    python process-digiteca-gemeni/process_digiteca_gemini_full.py --range 1-200

    # Process rows 500 through 1000
    python process-digiteca-gemeni/process_digiteca_gemini_full.py --range 500-1000
    ```
*   `--test`: Processes only the first 2 rows from the CSV. This overrides the `--range` argument if both are provided.
    ```bash
    python process-digiteca-gemeni/process_digiteca_gemini_full.py --test
    ```
2.  **Generate Viewer Index:** After processing, run the index generator script. This creates/updates the `digiteca-viewer/index.json` file, which the web viewer needs for navigation.
    ```bash
    # Navigate into the viewer directory first
    cd process-digiteca-gemeni/digiteca-viewer
    python generate_index.py
    cd ../.. # Go back to the original directory if needed
    ```
    *(Alternatively: `python process-digiteca-gemeni/digiteca-viewer/generate_index.py`)*

3.  **Run Web Viewer:** The easiest way to start the viewer is to use the provided script:
    ```bash
    # Navigate into the viewer directory
    cd process-digiteca-gemeni/digiteca-viewer
    # Make executable (if needed)
    chmod +x start_viewer.sh
    # Run the script
    ./start_viewer.sh
    ```
    This script automatically runs the index generator and then starts the required web server from the correct parent directory (`process-digiteca-gemeni`). It will output the URL to access the viewer (typically `http://<your-server-ip>:8000/digiteca-viewer/`). See the `digiteca-viewer/README.md` for more details.

## Processing Script Output

*   **Transcription JSON Files:** For each successfully processed image pair, a JSON file named `{Barcode}.json` is created in the `process-digiteca-gemeni/transcriptions/` directory. This file contains the source metadata (ID, filenames, etc.) and the structured transcription data returned by the Gemini API, along with API usage metadata.
*   **Console Output:** The script prints progress messages, including which barcode/row is being processed, success/failure status, retry attempts, and any errors encountered.
*   **Final Summary:** Upon completion, the processing script prints a JSON summary to the console, including:
    *   Total script runtime.
    *   Total time spent on API calls.
    *   Total estimated cost.
    *   Counts of successfully processed, skipped (already existing output), and failed images.
    *   Average cost and API time per processed image.
    *   Details of any failed items.

## Notes

*   The script includes a retry mechanism (up to 3 attempts with exponential backoff) for Gemini API calls.
*   It attempts to handle potential CSV encoding issues (tries UTF-8 then Latin-1) when reading the source CSV.
*   If an output JSON file for a specific barcode already exists in the `transcriptions` directory, the processing script will skip that row to allow for resuming interrupted runs.
*   The `generate_index.py` script must be run after processing to update the list of files available to the web viewer.
