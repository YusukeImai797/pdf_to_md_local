import requests
import json


class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234/v1", api_key=None):
        self.base_url = base_url
        self.api_key = api_key

    def chat_completion(self, messages, temperature=0.1, model=None, max_tokens=None, timeout=(20, 900)):
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "messages": messages,
            "temperature": temperature,
            **({"max_tokens": int(max_tokens)} if max_tokens is not None else {})
        }
        if model:
            data["model"] = model

        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def vision_request_multi(self, image_base64_list, prompt, model=None, temperature=0.1, max_tokens=900, timeout=(20, 900)):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        } for img_b64 in image_base64_list
                    ]
                ]
            }
        ]
        response = self.chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        return response["choices"][0]["message"]["content"]

    def vision_describe_images(self, prompt, image_base64_list, model=None, temperature=0.1, max_tokens=900, timeout=(20, 900)):
        """Compatibility method expected by the converter.

        Accepts (prompt, [b64,...]) and returns the assistant text.
        Also tolerates swapped positional order for older callers.
        """
        if isinstance(prompt, (list, tuple)) and isinstance(image_base64_list, str):
            prompt, image_base64_list = image_base64_list, prompt
        return self.vision_request_multi(image_base64_list, prompt, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    def check_connection(self):
        try:
            requests.get(f"{self.base_url}/models", timeout=5)
            return True
        except Exception:
            return False
