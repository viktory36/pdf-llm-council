"""PDF processing utilities for OpenRouter native PDF support."""

import base64
from typing import Dict, Any, List


def encode_pdf_to_base64(pdf_bytes: bytes) -> str:
    """
    Encode PDF bytes to base64 string for OpenRouter API.

    Args:
        pdf_bytes: PDF file content as bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(pdf_bytes).decode('utf-8')


def create_message_with_pdf(user_message: str, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Create OpenRouter API message with PDF content using native PDF support.
    
    OpenRouter supports PDFs in multimodal format similar to images.
    The PDF is encoded as base64 and sent as a content part.

    Args:
        user_message: The user's text message
        pdf_bytes: PDF file content as bytes
        filename: Name of the PDF file

    Returns:
        Message list formatted for OpenRouter API with PDF content
    """
    # Encode PDF to base64
    pdf_base64 = encode_pdf_to_base64(pdf_bytes)
    
    # Default message if user didn't provide one
    if not user_message.strip():
        user_message = "Please analyze this PDF document."
    
    # Create multimodal message content
    # Format: array of content parts with text and PDF data
    content = [
        {
            "type": "text",
            "text": user_message
        },
        {
            "type": "image_url",  # OpenRouter uses image_url type for PDFs too
            "image_url": {
                "url": f"data:application/pdf;base64,{pdf_base64}"
            }
        }
    ]
    
    return [{"role": "user", "content": content}]
