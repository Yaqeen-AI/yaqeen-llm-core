"""
Singleton LLM loader with Groq Cloud API support and local fallback.
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
    _DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_DIR, ".env"))
    load_dotenv(os.path.join(os.path.dirname(_DIR), ".env"))
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(_DIR)), ".env"))
except Exception:
    pass


class GroqLLM(Runnable):
    """
    Zero-dependency custom LangChain Runnable wrapper for Groq Cloud API.
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen/qwen3-32b",
        reasoning: bool = False
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.reasoning = reasoning
        
        override = os.getenv("GROQ_MODEL_OVERRIDE")
        if override and not reasoning:
            self.model_name = override

    def invoke(
        self,
        input_val: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> str:
        # Resolve prompt to string
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
            "temperature": 0.2,
        }

        # Qwen3-32b reasoning mode — no native 'thinking' parameter support on Groq yet
        if self.reasoning:
            # We just use a slightly higher temperature for better reasoning variance
            payload["temperature"] = 0.6

        print(f"   [LLM] -> Model: {self.model_name} | "
              f"Reasoning: {self.reasoning} | "
              f"Temp: {payload['temperature']}")

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60  # reasoning needs more time
            )
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            print(f"   [LLM] -> HTTP {response.status_code}: {response.text[:300]}")
            if self.reasoning:
                print("   [LLM] -> Retrying without reasoning temp...")
                payload["temperature"] = 0.2
                try:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json=payload, headers=headers, timeout=30
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except Exception as e2:
                    return f"[Groq API Error]: {e2}"
            return f"[Groq API Error]: {e}"

        except Exception as e:
            return f"[Groq API Error]: {e}"


_lock = threading.Lock()
_llm_instance = None
_tokenizer_instance = None

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def _init_local():
    """Load the local model (fallback if Groq key is absent)."""
    global _llm_instance, _tokenizer_instance
    print("[llm_loader] Initializing local model fallback...")

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
            best_dtype = (torch.bfloat16
                          if torch.cuda.is_bf16_supported()
                          else torch.float16)
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
                print(f"[llm_loader] 4-bit load failed ({e}), "
                      f"falling back to full-precision.")
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
            print(f"[llm_loader] LangChain pipeline import failed ({e}), "
                  f"using dummy LLM.")
            class _DummyLLM:
                def __call__(self, prompt):
                    return "[Dummy LLM] Model unavailable."
                def invoke(self, prompt):
                    return self(prompt)
            _llm_instance = _DummyLLM()

        print("[llm_loader] Local model ready.")

    except Exception as exc:
        print(f"[llm_loader] WARNING: Failed to load local model "
              f"({exc}); using dummy LLM.")
        class _DummyLLM:
            def __call__(self, prompt):
                return "[Dummy LLM] Model unavailable."
            def invoke(self, prompt):
                return self(prompt)
        _llm_instance = _DummyLLM()
        _tokenizer_instance = None


def get_llm(reasoning: bool = False):
    """
    Return LLM instance.
    reasoning=True  → Qwen3-32b with thinking mode enabled
    reasoning=False → Qwen3-32b standard mode (singleton)
    """
    global _llm_instance

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        if reasoning:
            # Always return a fresh reasoning instance — not cached
            print("[llm_loader] Returning Qwen3-32B with reasoning mode...")
            return GroqLLM(
                api_key=groq_key,
                model_name="qwen/qwen3-32b",
                reasoning=True
            )
        # Standard singleton
        if _llm_instance is None:
            with _lock:
                if _llm_instance is None:
                    print("[llm_loader] Initializing Qwen3-32B via "
                          "Groq Cloud API...")
                    _llm_instance = GroqLLM(
                        api_key=groq_key,
                        model_name="qwen/qwen3-32b",
                        reasoning=False
                    )
        return _llm_instance

    # No Groq key — use local
    if _llm_instance is None:
        with _lock:
            if _llm_instance is None:
                _init_local()
    return _llm_instance


class HeuristicTokenizer:
    """
    Zero-dependency heuristic tokenizer.
    Arabic chars: chunk size 2. English/other: chunk size 3.
    """
    def encode(self, text: str):
        if not text:
            return []
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            char = text[i]
            is_arabic = (
                '\u0600' <= char <= '\u06FF' or
                '\u0750' <= char <= '\u077F' or
                '\u08A0' <= char <= '\u08FF' or
                '\uFB50' <= char <= '\uFDFF' or
                '\uFE70' <= char <= '\uFEFF'
            )
            chunk_size = 2 if is_arabic else 3
            tokens.append(text[i:i + chunk_size])
            i += chunk_size
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


def get_tokenizer():
    """Return singleton tokenizer."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        with _lock:
            if _tokenizer_instance is None:
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    _tokenizer_instance = HeuristicTokenizer()
                else:
                    _init_local()
    return _tokenizer_instance