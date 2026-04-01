"""
Extended OpenAIService with support for arbitrary extra body parameters.

Both vllm and sglang expose OpenAI-compatible APIs but accept additional
non-standard request body fields (sampling params, constrained decoding
backends, chat template kwargs, etc.).  The OpenAI Python SDK forwards
anything in ``extra_body`` directly into the HTTP request body, so this
subclass exposes a single ``openai_extra_body`` dict that is passed through
on every inference call.

Usage via handler ``llm_config``:

    {
        "llm_service": "openai_service.OpenAIServiceWithExtraBody",
        "llm_config": {
            "openai_base_url": "http://localhost:8000/v1",
            "openai_api_key": "EMPTY",
            "openai_model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "openai_extra_body": {
                "chat_template_kwargs": {"enable_thinking": false},
                "top_k": 20,
                "min_p": 0.05,
                "guided_decoding_backend": "xgrammar"
            }
        }
    }

Common extra parameters by backend
-----------------------------------
vllm:
    Sampling       : top_k, min_p, repetition_penalty, length_penalty,
                     best_of, use_beam_search, ignore_eos, min_tokens,
                     stop_token_ids, allowed_token_ids
    Output control : skip_special_tokens, spaces_between_special_tokens,
                     truncate_prompt_tokens, prompt_logprobs,
                     include_stop_str_in_output, echo, return_token_ids
    Structured out : structured_outputs (new, v0.12+),
                     guided_json / guided_regex / guided_choice /
                     guided_grammar / guided_decoding_backend (deprecated v0.12+)
    Chat template  : chat_template, chat_template_kwargs,
                     add_generation_prompt, continue_final_message
    Advanced       : documents, priority, cache_salt, vllm_xargs

sglang:
    Sampling       : top_k, min_p, min_new_tokens (alias: min_tokens),
                     repetition_penalty, ignore_eos, skip_special_tokens,
                     spaces_between_special_tokens
    Structured out : regex, ebnf,
                     guided_decoding_backend (xgrammar / outlines / llguidance)
    Chat template  : chat_template_kwargs  e.g. {"enable_thinking": true}
    Log probs      : return_text_in_logprobs, logprob_start_len
    Advanced       : separate_reasoning, return_hidden_states,
                     return_routed_experts, priority, cache_salt,
                     rid, lora_path
"""

import json
import time
from typing import Annotated, Any, Dict, List

import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services.openai import OpenAIService

logger = get_logger()


class OpenAIServiceWithExtraBody(OpenAIService):
    openai_extra_body: Annotated[
        Dict[str, Any],
        "Extra fields merged into the request body on every call. "
        "Use this to pass backend-specific parameters supported by vllm or sglang "
        "(e.g. top_k, min_p, chat_template_kwargs, guided_decoding_backend, regex, ebnf).",
    ] = {}

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        extra_body = self.openai_extra_body

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.beta.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                        "HTTP-Referer": "https://github.com/datalab-to/marker",
                    },
                    extra_body=extra_body,
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                response_text = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                if tries == total_tries:
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                    )
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"OpenAI inference failed: {e}")
                break

        return {}
