import json
import re
import os

def convert_to_valid_json():
    """
    Reads a text file containing a JSON-like structure with comments,
    removes the comments, parses it into valid JSON, and writes it
    to an output file, overwriting if it exists.
    Uses hardcoded file paths.
    """
    input_file_path = os.path.join(os.path.dirname(__file__), 'prompt_output_structure.txt')
    output_file_path = os.path.join(os.path.dirname(__file__), 'prompt_validation.json')

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            content = infile.read()

        # Remove // comments
        content_no_comments = re.sub(r"//.*", "", content)

        # Remove trailing commas before closing braces and brackets
        content_no_trailing_commas = re.sub(r",\s*([\}\]])", r"\1", content_no_comments)

        # Parse the cleaned JSON string
        data = json.loads(content_no_trailing_commas)

        # Write valid JSON to output file, overwriting if it exists
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=2, ensure_ascii=False)

        print(f"Successfully converted '{os.path.basename(input_file_path)}' to '{os.path.basename(output_file_path)}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from input file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    convert_to_valid_json()
