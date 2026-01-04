"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content' (and optional 'attachments')
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Process messages to handle attachments (multimodal)
    processed_messages = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        attachments = msg.get("attachments", [])
        
        if not attachments:
            new_msg["content"] = msg["content"]
        else:
            # Multimodal content
            content_parts = []
            
            # Add text part
            if msg["content"]:
                content_parts.append({
                    "type": "text",
                    "text": msg["content"]
                })
            
            # Add attachment parts
            for attachment in attachments:
                # Expecting attachment to have 'type' (mime type) and 'base64' (data)
                mime_type = attachment.get("type", "application/pdf")
                base64_data = attachment.get("base64", "")
                
                # OpenRouter (and many providers) handle PDFs via image_url with data URI
                # or specialized 'image_url' object.
                # For PDF specifically, many providers on OpenRouter support it via standard image_url
                # with the correct mime type in the data URI.
                data_uri = f"data:{mime_type};base64,{base64_data}"
                
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri
                    }
                })
            
            new_msg["content"] = content_parts
            
        processed_messages.append(new_msg)

    payload = {
        "model": model,
        "messages": processed_messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error response from OpenRouter: {response.text}")
                
            response.raise_for_status()

            data = response.json()
            if 'choices' not in data or not data['choices']:
                print(f"No choices in response: {data}")
                return None
                
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
