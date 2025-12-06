#!/usr/bin/env python3
"""
Judge evaluation using OpenAI APIs.

Supports both the Chat Completions API (for non-reasoning models) and the
Responses API (for GPT-5 and other reasoning models).

Handles batch creation, submission, monitoring, and result parsing.
Uses Batch API for 50% cost savings compared to standard API.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI


class JudgeEvaluator:
    """
    Evaluates model responses using OpenAI judge models via Batch API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_model: str = "gpt-5-nano",
        temperature: float = 0.0,
        max_tokens: int = 10,
        reasoning_effort: str = "low",
        use_batch_api: bool = True
    ):
        """
        Initialize judge evaluator.

        Args:
            api_key: OpenAI API key (or None to use OPENAI_API_KEY env var)
            judge_model: Judge model name (default: gpt-5-nano)
            temperature: Judge temperature (default: 0.0 for deterministic)
            max_tokens: Max tokens per judge response (default: 10)
            reasoning_effort: Reasoning effort for gpt-5 models (low/medium/high)
            use_batch_api: Whether to use Batch API (default: True)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key, timeout=300.0)  # 5 minute timeout to prevent hanging

        self.judge_model = judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.use_batch_api = use_batch_api

    def create_judge_requests(
        self,
        responses: List[Dict[str, Any]],
        judge_prompts: Dict[str, callable],
        evaluation_type: str,
        skip_coherency: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create judge API requests from model responses.

        Args:
            responses: List of model response records
            judge_prompts: Dictionary of judge functions (from judges.py)
            evaluation_type: "em" or "political"
            skip_coherency: Whether to skip coherency judge (default: False)

        Returns:
            List of API request dictionaries
        """
        requests = []

        for response_record in responses:
            question = response_record['question_text']
            response = response_record['response']
            prompt_slant = response_record.get('prompt_slant')

            # Skip error responses
            if response.startswith("[ERROR:"):
                continue

            # Generate judge prompts based on evaluation type
            for judge_name, judge_func in judge_prompts.items():
                # Skip coherency judge if requested
                if skip_coherency and judge_name == 'coherency':
                    continue

                # Some judges need prompt_slant parameter
                if judge_name in ['stance_bias', 'escalation'] and prompt_slant:
                    judge_prompt = judge_func(question, response, prompt_slant)
                else:
                    judge_prompt = judge_func(question, response)

                # Build request body based on model type
                is_gpt5 = "gpt-5" in self.judge_model.lower()

                if is_gpt5:
                    # For gpt-5 models, use Responses API format
                    body = {
                        "model": self.judge_model,
                        "input": judge_prompt,
                        "max_output_tokens": self.max_tokens
                    }
                    # Only add reasoning if effort is not 'none'
                    if self.reasoning_effort and self.reasoning_effort.lower() != 'none':
                        body["reasoning"] = {"effort": self.reasoning_effort}
                    endpoint = "/v1/responses"
                else:
                    # For non-reasoning models, use Chat Completions API
                    body = {
                        "model": self.judge_model,
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "max_completion_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                    endpoint = "/v1/chat/completions"

                request = {
                    "custom_id": f"{response_record['model_name']}_{response_record['question_id']}_s{response_record['sample_idx']}_{judge_name}",
                    "method": "POST",
                    "url": endpoint,
                    "body": body,
                    # Store metadata for parsing results
                    "metadata": {
                        "model_name": response_record['model_name'],
                        "question_id": response_record['question_id'],
                        "sample_idx": response_record['sample_idx'],
                        "judge_name": judge_name,
                        "evaluation_type": evaluation_type,
                        "question_text": question,
                        "response_text": response,
                        "prompt_slant": prompt_slant
                    }
                }

                requests.append(request)

        return requests

    def save_batch_requests(
        self,
        requests: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Save batch requests to JSONL file.

        Args:
            requests: List of API request dictionaries
            output_path: Path to save JSONL file

        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for request in requests:
                # Don't include metadata in the actual batch request
                batch_request = {
                    "custom_id": request["custom_id"],
                    "method": request["method"],
                    "url": request["url"],
                    "body": request["body"]
                }
                f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(requests)} batch requests to {output_file}")
        return str(output_file)

    def submit_batch(self, batch_file_path: str, endpoint: str = None) -> str:
        """
        Submit batch file to OpenAI Batch API.

        Args:
            batch_file_path: Path to JSONL batch file
            endpoint: API endpoint (auto-detected from batch file if None)

        Returns:
            Batch ID
        """
        print(f"\nSubmitting batch to OpenAI API...")

        # Auto-detect endpoint if not provided
        if endpoint is None:
            # Check first request to determine endpoint
            with open(batch_file_path, 'r') as f:
                first_request = json.loads(f.readline())
                endpoint = first_request.get('url', '/v1/chat/completions')
            print(f"Auto-detected endpoint: {endpoint}")

        # Upload file
        with open(batch_file_path, 'rb') as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )

        print(f"✓ Uploaded batch file: {batch_input_file.id}")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={
                "description": f"ALIGNROT evaluation - {datetime.now().isoformat()}"
            }
        )

        print(f"✓ Batch created: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Total requests: {batch.request_counts.total if hasattr(batch, 'request_counts') else 'Unknown'}")

        return batch.id

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Check status of a batch.

        Args:
            batch_id: Batch ID

        Returns:
            Batch status dictionary
        """
        batch = self.client.batches.retrieve(batch_id)

        status = {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": getattr(batch, 'completed_at', None),
            "failed_at": getattr(batch, 'failed_at', None),
            "errors": getattr(batch, 'errors', None)
        }

        if hasattr(batch, 'request_counts'):
            status["request_counts"] = {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed
            }

        return status

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """
        Wait for batch to complete, polling periodically.

        Args:
            batch_id: Batch ID
            poll_interval: Seconds between status checks

        Returns:
            Final batch status
        """
        print(f"\nWaiting for batch {batch_id} to complete...")
        print(f"(Polling every {poll_interval} seconds)")

        while True:
            status = self.check_batch_status(batch_id)

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch status: {status['status']}")
            if 'request_counts' in status:
                counts = status['request_counts']
                print(f"  Completed: {counts['completed']}/{counts['total']}")
                print(f"  Failed: {counts['failed']}/{counts['total']}")

            if status['status'] in ['completed', 'failed', 'expired', 'cancelled']:
                print(f"\n✓ Batch finished with status: {status['status']}")
                return status

            time.sleep(poll_interval)

    def download_batch_results(self, batch_id: str, output_path: str) -> str:
        """
        Download batch results to file.

        Args:
            batch_id: Batch ID
            output_path: Path to save results JSONL

        Returns:
            Path to saved file
        """
        print(f"\nDownloading batch results...")

        batch = self.client.batches.retrieve(batch_id)

        if batch.status != 'completed':
            raise RuntimeError(f"Batch not completed. Status: {batch.status}")

        # Download output file
        output_file_id = batch.output_file_id
        file_content = self.client.files.content(output_file_id).read()

        # Save to disk
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            f.write(file_content)

        print(f"✓ Downloaded batch results to {output_file}")

        # Also download error file if exists
        if hasattr(batch, 'error_file_id') and batch.error_file_id:
            error_content = self.client.files.content(batch.error_file_id).read()
            error_file = output_file.parent / f"{output_file.stem}_errors.jsonl"
            with open(error_file, 'wb') as f:
                f.write(error_content)
            print(f"✓ Downloaded error file to {error_file}")

        return str(output_file)

    def parse_batch_results(
        self,
        results_file: str,
        requests_metadata: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Parse batch results and combine with original metadata.

        Args:
            results_file: Path to batch results JSONL
            requests_metadata: Original requests with metadata
            output_path: Path to save parsed evaluations

        Returns:
            List of evaluation records
        """
        print(f"\nParsing batch results...")

        # Load results
        results = {}
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                custom_id = result['custom_id']
                results[custom_id] = result

        # Combine with metadata
        evaluations = []

        for request in requests_metadata:
            custom_id = request['custom_id']
            metadata = request['metadata']

            if custom_id not in results:
                print(f"⚠ Warning: Missing result for {custom_id}")
                continue

            result = results[custom_id]

            # Extract score from response (handle both Chat Completions and Responses API formats)
            try:
                response_body = result['response']['body']

                # Try Responses API format first (output_text field)
                if 'output_text' in response_body:
                    raw_response = response_body['output_text'].strip()
                # Fall back to Chat Completions format
                elif 'choices' in response_body:
                    raw_response = response_body['choices'][0]['message']['content'].strip()
                else:
                    raise KeyError("No output_text or choices found in response")

                score = int(raw_response)

                if not (0 <= score <= 100):
                    print(f"⚠ Warning: Score {score} out of range for {custom_id}")

            except (ValueError, KeyError, IndexError) as e:
                print(f"⚠ Warning: Failed to parse score for {custom_id}: {e}")
                raw_response = str(result.get('response', {}).get('body', 'N/A'))
                score = None

            evaluation = {
                "model_name": metadata['model_name'],
                "question_id": metadata['question_id'],
                "sample_idx": metadata['sample_idx'],
                "judge_name": metadata['judge_name'],
                "evaluation_type": metadata['evaluation_type'],
                "question_text": metadata['question_text'],
                "response_text": metadata['response_text'],
                "prompt_slant": metadata.get('prompt_slant'),
                "score": score,
                "raw_response": raw_response,
                "timestamp": datetime.now().isoformat()
            }

            evaluations.append(evaluation)

        # Save evaluations
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for evaluation in evaluations:
                f.write(json.dumps(evaluation, ensure_ascii=False) + '\n')

        print(f"✓ Parsed {len(evaluations)} evaluations")
        print(f"✓ Saved to {output_file}")

        # Print summary
        valid_scores = [e for e in evaluations if e['score'] is not None]
        print(f"\nSummary:")
        print(f"  Total evaluations: {len(evaluations)}")
        print(f"  Valid scores: {len(valid_scores)}")
        print(f"  Failed parses: {len(evaluations) - len(valid_scores)}")

        return evaluations

    def evaluate_synchronous(
        self,
        requests: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate using synchronous API calls (for debugging/testing).

        Args:
            requests: List of API request dictionaries
            output_path: Path to save evaluations

        Returns:
            List of evaluation records
        """
        print(f"\nRunning synchronous evaluation (non-batch mode)")
        print(f"Total requests: {len(requests)}")

        evaluations = []

        for idx, request in enumerate(requests, 1):
            print(f"[{idx}/{len(requests)}] {request['custom_id']}...", end=" ", flush=True)

            try:
                is_gpt5 = "gpt-5" in request['body']['model'].lower()

                # Use the appropriate API based on model type
                if is_gpt5:
                    # Use Responses API for GPT-5 models
                    api_params = {
                        "model": request['body']['model'],
                        "input": request['body']['input'],
                        "max_output_tokens": request['body'].get('max_output_tokens', 500)
                    }

                    # Add reasoning if present and not 'none'
                    if 'reasoning' in request['body']:
                        api_params['reasoning'] = request['body']['reasoning']

                    response = self.client.responses.create(**api_params)

                    # Use output_text attribute which contains the final text output
                    raw_response = response.output_text.strip() if response.output_text else ""

                else:
                    # Use Chat Completions API for non-reasoning models
                    api_params = {
                        "model": request['body']['model'],
                        "messages": request['body']['messages'],
                        "max_completion_tokens": request['body'].get('max_completion_tokens', 500)
                    }

                    # Add temperature if present
                    if 'temperature' in request['body']:
                        api_params['temperature'] = request['body']['temperature']

                    response = self.client.chat.completions.create(**api_params)
                    raw_response = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

                # Try to parse score
                try:
                    score = int(raw_response)
                    if not (0 <= score <= 100):
                        print(f"⚠ Score {score} out of range")
                except ValueError:
                    score = None
                    # Show full response for debugging
                    if len(raw_response) == 0:
                        if is_gpt5:
                            # Check if there's actually content in the response
                            if hasattr(response, 'status'):
                                print(f"⚠ Empty response (status: {response.status})")
                            else:
                                print(f"⚠ Empty response")
                        else:
                            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                            print(f"⚠ Empty response (finish_reason: {finish_reason})")
                    else:
                        # Show first 100 chars of invalid response for debugging
                        preview = raw_response[:100] + "..." if len(raw_response) > 100 else raw_response
                        print(f"⚠ Invalid: '{preview}'")

                metadata = request['metadata']

                # Get finish reason based on model type
                if is_gpt5:
                    # Responses API doesn't have finish_reason in the same way
                    finish_reason = getattr(response, 'finish_reason', None)
                else:
                    finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None

                evaluation = {
                    "model_name": metadata['model_name'],
                    "question_id": metadata['question_id'],
                    "sample_idx": metadata['sample_idx'],
                    "judge_name": metadata['judge_name'],
                    "evaluation_type": metadata['evaluation_type'],
                    "question_text": metadata['question_text'],
                    "response_text": metadata['response_text'],
                    "prompt_slant": metadata.get('prompt_slant'),
                    "score": score,
                    "raw_response": raw_response,
                    "finish_reason": finish_reason,
                    "timestamp": datetime.now().isoformat()
                }

                evaluations.append(evaluation)
                if score is not None:
                    print(f"✓ Score: {score}")
                else:
                    print(f"✗ Failed to parse")

            except Exception as e:
                print(f"✗ Error: {e}")
                metadata = request['metadata']
                evaluation = {
                    "model_name": metadata['model_name'],
                    "question_id": metadata['question_id'],
                    "sample_idx": metadata['sample_idx'],
                    "judge_name": metadata['judge_name'],
                    "evaluation_type": metadata['evaluation_type'],
                    "score": None,
                    "raw_response": f"[ERROR: {str(e)}]",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                evaluations.append(evaluation)

        # Save evaluations
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for evaluation in evaluations:
                f.write(json.dumps(evaluation, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved {len(evaluations)} evaluations to {output_file}")

        return evaluations
