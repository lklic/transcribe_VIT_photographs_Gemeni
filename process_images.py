import sys
import json
import time
import base64
import httpx
from pathlib import Path
from typing import Dict, Any, List
import io
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
from models_config import ProcessingConfig, MODEL_CONFIGS

class BenchmarkStats:
    def __init__(self):
        self.total_cost = 0.0
        self.total_time = 0
        self.total_tokens = 0
        self.processed_images = 0
        self.failed_images = []

    def update(self, result: Dict[str, Any]):
        self.total_cost += result['cost']
        self.total_time += result['request_time']
        self.total_tokens += result['total_tokens']
        self.processed_images += 1

    def add_failure(self, image_id: str, error: str):
        self.failed_images.append({"image_id": image_id, "error": str(error)})

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 4),
            "total_time": self.total_time,
            "total_tokens": self.total_tokens,
            "processed_images": self.processed_images,
            "average_cost_per_image": round(self.total_cost / max(1, self.processed_images), 4),
            "average_time_per_image": round(self.total_time / max(1, self.processed_images), 2),
            "failed_images": self.failed_images
        }

class ImageProcessor:
    def __init__(self, config: ProcessingConfig, prompt_file: str):
        self.config = config
        with open(prompt_file, "r") as f:
            self.prompt = f.read()
        self._setup_client()
        
    def _setup_client(self):
        if self.config.api_type == 'openai':
            with open("key.secret", "r") as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
        elif self.config.api_type == 'claude':
            with open("claudekey.secret", "r") as f:
                api_key = f.read().strip()
            self.client = Anthropic(api_key=api_key)
        elif self.config.api_type == 'gemini':
            with open("geminikey.secret", "r") as f:
                api_key = f.read().strip()
            genai.configure(api_key=api_key)
            # Safety settings can be adjusted if needed
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            self.client = genai.GenerativeModel(
                self.config.model,
                safety_settings=safety_settings
            )
        else:
            raise ValueError(f"Unsupported api_type: {self.config.api_type}")

    def _get_base64_image(self, url: str) -> str:
        response = httpx.get(url)
        return base64.b64encode(response.content).decode("utf-8")

    def process_images(self, image_id: str) -> Dict[str, Any]:
        photo_id = image_id.split('!')[1]
        base_url = "https://iiif.itatti.harvard.edu/iiif/2/digiteca!{}_{:d}.jpg/full/1024,1024/0/default.jpg"
        image_url_1 = base_url.format(image_id, 1)
        image_url_2 = base_url.format(image_id, 2)

        start_time = time.time()
        
        for attempt in range(2):
            try:
                if self.config.api_type == 'openai':
                    response = self._process_openai(image_url_1, image_url_2)
                elif self.config.api_type == 'claude':
                    response = self._process_claude(image_url_1, image_url_2)
                elif self.config.api_type == 'gemini':
                    response = self._process_gemini(image_url_1, image_url_2)
                else:
                    raise ValueError(f"Unknown api_type: {self.config.api_type}")

                elapsed_time = round(time.time() - start_time)
                return self._format_output(response, photo_id, elapsed_time)
            except Exception as e:
                if attempt == 0:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise

    def _process_openai(self, url1: str, url2: str):
        response_params = {
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt + "\n\nPlease provide the result in a JSON format."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url1,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url2,
                        },
                    }
                ]
            }],
            "response_format": {"type": "json_object"}
        }
        
        return self.client.chat.completions.create(**response_params)

    def _process_claude(self, url1: str, url2: str):
        img1_data = self._get_base64_image(url1)
        img2_data = self._get_base64_image(url2)
        
        return self.client.messages.create(
            model=self.config.model,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt + "\n\nPlease provide the result in a JSON format."},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img1_data
                    }},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img2_data
                    }}
                ]
            }]
        )

    def _process_gemini(self, url1: str, url2: str):
        # Fetch image bytes
        img1_bytes = httpx.get(url1).content
        img2_bytes = httpx.get(url2).content

        # Prepare image parts for Gemini API
        img1_part = {"mime_type": "image/jpeg", "data": img1_bytes}
        img2_part = {"mime_type": "image/jpeg", "data": img2_bytes}

        prompt_parts = [
            self.prompt + "\n\nPlease provide the result in a JSON format.",
            img1_part,
            img2_part
        ]

        # Specify JSON output format
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )

        return self.client.generate_content(
            prompt_parts,
            generation_config=generation_config
        )

    def _format_output(self, response, photo_id: str, request_time: int) -> Dict[str, Any]:
        if self.config.api_type == 'openai':
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            content = response.choices[0].message.content
            total_tokens = input_tokens + output_tokens
        elif self.config.api_type == 'claude':
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            content = response.content[0].text
            total_tokens = input_tokens + output_tokens
        elif self.config.api_type == 'gemini':
            # Gemini token counts might be in usage_metadata
            # Note: Gemini API might not always return token counts reliably for multimodal inputs yet.
            # Using 0 as fallback if not present.
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            content = response.text
            total_tokens = getattr(response.usage_metadata, 'total_token_count', input_tokens + output_tokens) # Use total if available
        else:
             raise ValueError(f"Unknown api_type for formatting: {self.config.api_type}")

        input_cost = (input_tokens / 1_000_000) * self.config.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.config.output_cost_per_million

        # Handle markdown code blocks in Claude 3.7 output
        if '```json' in content:
            # Extract JSON from markdown code block
            json_start = content.find('```json\n') + 8
            json_end = content.rfind('```')
            if json_end > json_start:
                content = content[json_start:json_end].strip()

        try:
            annotations = json.loads(content)
        except json.JSONDecodeError:
            annotations = content

        return {
            "photo_id": photo_id,
            "model": self.config.model,
            "annotations": annotations,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": input_cost + output_cost,
            "status": "OK",
            "request_time": request_time
        }

def run_benchmark(image_ids: List[str], prompt_file: str, models: List[str] = None):
    output_dir = Path("benchmark_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create directories for each model
    for model in MODEL_CONFIGS.keys():
        model_dir = output_dir / model
        model_dir.mkdir(exist_ok=True)
    
    # Filter models if specified
    models_to_run = {k: v for k, v in MODEL_CONFIGS.items() if models is None or k in models}
    
    # Check if benchmark summary already exists
    summary_file = output_dir / "benchmark_summary.json"
    existing_benchmark_results = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                existing_benchmark_results = json.load(f)
            print(f"Loaded existing benchmark summary with {len(existing_benchmark_results)} models")
        except json.JSONDecodeError:
            print("Warning: Existing benchmark summary is invalid, creating new one")
    
    # Process each model sequentially
    benchmark_results = existing_benchmark_results.copy()
    for model_name, config in models_to_run.items():
        print(f"\nProcessing with {model_name}...")
        processor = ImageProcessor(config, prompt_file)
        stats = BenchmarkStats()
        
        for image_id in image_ids:
            try:
                print(f"Processing {image_id} with {model_name}...")
                result = processor.process_images(image_id)
                
                # Save individual result
                output_file = output_dir / model_name / f"{image_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                stats.update(result)
                print(f"Successfully processed {image_id}")
                
            except Exception as e:
                print(f"Failed to process {image_id} with {model_name}: {str(e)}")
                stats.add_failure(image_id, e)
        
        # Save model statistics
        benchmark_results[model_name] = stats.get_summary()
    
    # Save overall benchmark results
    with open(output_dir / "benchmark_summary.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Updated benchmark summary with {len(benchmark_results)} models")
    
    return benchmark_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 process_images.py <mode> [args...]")
        print("Modes:")
        print("  single <model> <image_id> <prompt_file>")
        print("  benchmark <prompt_file>")
        sys.exit(1)

    mode = sys.argv[1]
    
    if mode == "single":
        if len(sys.argv) != 5:
            print("Usage: python3 process_images.py single <model> <image_id> <prompt_file>")
            sys.exit(1)
            
        model = sys.argv[2]
        if model not in MODEL_CONFIGS:
            print(f"Invalid model. Choose from: {', '.join(MODEL_CONFIGS.keys())}")
            sys.exit(1)
            
        image_id = sys.argv[3].replace('"', '')
        prompt_file = sys.argv[4]

        # Process the single image
        processor = ImageProcessor(MODEL_CONFIGS[model], prompt_file)
        result = processor.process_images(image_id)
        
        # Save the result to the model's directory
        output_dir = Path("benchmark_data")
        output_dir.mkdir(exist_ok=True)
        model_dir = output_dir / model
        model_dir.mkdir(exist_ok=True)
        
        output_file = model_dir / f"{image_id}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update the benchmark summary with this result
        summary_file = output_dir / "benchmark_summary.json"
        benchmark_results = {}
        
        # Load existing summary if it exists
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    benchmark_results = json.load(f)
                print(f"Loaded existing benchmark summary with {len(benchmark_results)} models")
            except json.JSONDecodeError:
                print("Warning: Existing benchmark summary is invalid, creating new one")
        
        # Create or update stats for this model
        stats = BenchmarkStats()
        stats.update(result)
        benchmark_results[model] = stats.get_summary()
        
        # Save updated benchmark summary
        with open(summary_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"Updated benchmark summary for model {model}")
        print(json.dumps(result, indent=4))
        
    elif mode == "benchmark":
        if len(sys.argv) < 3:
            print("Usage: python3 process_images.py benchmark <prompt_file> [model1 model2 ...]")
            sys.exit(1)
            
        prompt_file = sys.argv[2]
        models = sys.argv[3:] if len(sys.argv) > 3 else None
        
        if models:
            invalid_models = [m for m in models if m not in MODEL_CONFIGS]
            if invalid_models:
                print(f"Invalid models: {', '.join(invalid_models)}")
                print(f"Choose from: {', '.join(MODEL_CONFIGS.keys())}")
                sys.exit(1)
        
        # Read image IDs from test-images.md
        with open("test-images.md", 'r') as f:
            image_ids = [line.strip() for line in f if line.strip()]
        
        # Run benchmark
        benchmark_results = run_benchmark(image_ids, prompt_file, models)
        
        # Print summary
        print("\nBenchmark Summary:")
        print(json.dumps(benchmark_results, indent=2))
        
    else:
        print(f"Invalid mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
