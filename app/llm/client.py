"""Local Ollama REST client; no cloud account needed. HTTP requests are opt-in only."""
import base64
import json
from pathlib import Path
import httpx
from app.config import settings

class OllamaError(RuntimeError): pass

def available():
    try:
        r=httpx.get(settings()['ollama_url'].rstrip('/')+'/api/tags',timeout=3)
        return r.status_code==200
    except (httpx.HTTPError, ValueError): return False

def chat(system: str, prompt: str, images: list[bytes]|None=None, json_output=True) -> str:
    conf=settings()
    msg={'role':'user','content':prompt}
    if images: msg['images']=[base64.b64encode(image).decode('ascii') for image in images]
    payload={'model':conf['ollama_model'],'stream':False,'messages':[{'role':'system','content':system},msg], 'options':{'temperature':0}}
    if json_output: payload['format']='json'
    try:
        response=httpx.post(conf['ollama_url'].rstrip('/')+'/api/chat',json=payload,timeout=conf['ollama_timeout'])
        response.raise_for_status()
        return response.json()['message']['content']
    except (httpx.HTTPError, KeyError, ValueError) as exc: raise OllamaError(f'Ollama unavailable or model failed: {type(exc).__name__}') from exc

def json_chat(system,prompt,images=None):
    content=chat(system,prompt,images,True)
    try: return json.loads(content)
    except json.JSONDecodeError as exc: raise OllamaError('Model response is not valid JSON') from exc
