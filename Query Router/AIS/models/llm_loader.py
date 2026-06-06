"""
Singleton LLM loader with Groq Cloud API support and local fallback.

Provides get_llm() and get_tokenizer() — thread-safe, lazy-initialized.
If GROQ_API_KEY is found in the environment, it uses Groq Cloud API (qwen/qwen3-32b) via a Runnable wrapper,
avoiding all torch/transformers imports to bypass Windows DLL load issues.
Otherwise, it lazy-loads the local Qwen model.
"""

import os
import warnings
import threading
import requests
from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable

# ── Load .env ──
try:
    from dotenv import load_dotenv
    # Search upwards for .env
    _DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_DIR, ".env"))
    load_dotenv(os.path.join(os.path.dirname(_DIR), ".env"))
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(_DIR)), ".env"))
except Exception:
    pass

class GroqLLM(Runnable):
    """
    Zero-dependency custom LangChain Runnable wrapper for Groq Cloud API.
    Bypasses importing langchain_core.language_models to avoid torch DLL issues.
    """
    def __init__(self, api_key: str, model_name: str = "qwen/qwen3-32b"):
        self.api_key = api_key
        self.model_name = model_name

    def invoke(self, input_val: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        # Resolve prompt value to string
        if hasattr(input_val, "to_string"):
            prompt_str = input_val.to_string()
        elif hasattr(input_val, "content"):
            prompt_str = input_val.content
        elif isinstance(input_val, dict):
            prompt_str = str(input_val)
        else:
            prompt_str = str(input_val)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_str}],
            "temperature": 0.2
        }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Groq API Error]: {e}"


_lock = threading.Lock()
_llm_instance = None
_tokenizer_instance = None

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def _init_local():
    """Load the local model (fallback option if Groq key is absent)."""
    global _llm_instance, _tokenizer_instance
    print("[llm_loader] Initializing local model fallback...")
    
    # Suppress warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

    try:
        import torch
    except Exception:
        class _DummyCuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def is_bf16_supported(): return False
        class _DummyTorch:
            cuda = _DummyCuda()
            float32 = None
        torch = _DummyTorch()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from huggingface_hub import login
        
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(hf_token, add_to_git_credential=False)
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            best_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=best_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=best_dtype,
                    attn_implementation="sdpa",
                )
            except Exception as e:
                print(f"[llm_loader] 4-bit load failed ({e}), falling back to full-precision.")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=best_dtype,
                    device_map="auto",
                    attn_implementation="sdpa",
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu",
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            batch_size=1,
        )
        _tokenizer_instance = tokenizer

        try:
            from langchain_huggingface import HuggingFacePipeline
            _llm_instance = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"[llm_loader] LangChain pipeline import failed ({e}), using dummy LLM.")
            class _DummyLLM:
                def __call__(self, prompt): return "[Dummy LLM] Model unavailable."
                def invoke(self, prompt): return self(prompt)
            _llm_instance = _DummyLLM()

        print(f"[llm_loader] Local model ready.")
    except Exception as exc:
        print(f"[llm_loader] WARNING: Failed to load local model ({exc}); using dummy LLM.")
        class _DummyLLM:
            def __call__(self, prompt): return "[Dummy LLM] Model unavailable."
            def invoke(self, prompt): return self(prompt)
        _llm_instance = _DummyLLM()
        _tokenizer_instance = None


def get_llm():
    """Return the singleton LLM."""
    global _llm_instance
    if _llm_instance is None:
        with _lock:
            if _llm_instance is None:
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    print("[llm_loader] Initializing Qwen-32B via Groq Cloud API...")
                    _llm_instance = GroqLLM(api_key=groq_key)
                else:
                    _init_local()
    return _llm_instance


def get_tokenizer():
    """Return the singleton tokenizer."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        with _lock:
            if _tokenizer_instance is None:
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    _tokenizer_instance = None
                else:
                    _init_local()
    return _tokenizer_instance