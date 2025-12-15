"""MLLM Client using OpenAI SDK for Qwen3-VL or compatible models"""
import base64
import os
from typing import Any, Dict, List, Optional
from openai import OpenAI


class MLLMClient:
    """Client for calling MLLM (Qwen3-VL) using OpenAI SDK"""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_tokens: int = 4096
    ):
        """
        Initialize MLLM Client
        
        Args:
            api_base: Base URL for API (e.g., http://localhost:8000/v1)
            api_key: API authentication key
            model: Model name
            max_tokens: Maximum tokens to generate
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
    
    def _image_to_base64(self, image_path: str) -> tuple[str, str]:
        """Convert image file to base64 string and get MIME type"""
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        with open(image_path, 'rb') as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')
            return base64_data, mime_type
    
    def _process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process messages to convert image paths to base64"""
        processed_messages = []
        
        for message in messages:
            processed_message = message.copy()
            
            if message.get('role') == 'user' and 'content' in message:
                processed_content = []
                
                for content_item in message['content']:
                    if isinstance(content_item, dict) and content_item.get('type') == 'image':
                        # Convert image path to base64
                        image_path = content_item['image']
                        base64_image, mime_type = self._image_to_base64(image_path)
                        
                        # Add as image_url format
                        processed_content.append({
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{mime_type};base64,{base64_image}',
                                'detail': 'high'
                            }
                        })
                    
                    else:
                        processed_content.append(content_item)
                
                processed_message['content'] = processed_content
            
            processed_messages.append(processed_message)
        
        return processed_messages
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """
        Generate response from MLLM
        
        Args:
            messages: List of message dicts with role and content
            max_tokens: Override default max_tokens
            temperature: Sampling temperature
            stream: Whether to stream response
        
        Returns:
            Generator yielding chunks if stream=True, or complete text if stream=False
        """
        # Process messages (convert image paths to base64)
        processed_messages = self._process_messages(messages)
        
        print(f"🔍 Calling MLLM model {self.model}...")
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=processed_messages,
            max_completion_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
            stream=stream
        )
        
        if stream:
            # Return generator for streaming
            def stream_generator():
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content
            return stream_generator()
        else:
            # Extract complete response
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                print(f"✅ MLLM response received ({len(generated_text)} chars)")
                return generated_text
            else:
                print(f"⚠️  Unexpected response format: {response}")
                return None
    
    def update_config(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """Update client configuration"""
        if api_base:
            self.api_base = api_base
        if api_key:
            self.api_key = api_key
        if model:
            self.model = model
        if max_tokens:
            self.max_tokens = max_tokens
        
        # Reinitialize client if base URL or key changed
        if api_base or api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
