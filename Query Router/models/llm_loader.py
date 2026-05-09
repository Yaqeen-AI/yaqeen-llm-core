"""
Singleton LLM loader with 4-bit quantization.

Provides get_llm() and get_tokenizer() — thread-safe, lazy-initialized.
The model is loaded once on first access and reused across all modules.
"""

import os
import warnings
import threading

# ── Set HF_HOME *before* importing transformers / huggingface_hub ──
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_PATH)

_hf_home = os.getenv("HF_HOME")
if _hf_home:
    os.environ["HF_HOME"] = _hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")

# Suppress noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login

_lock = threading.Lock()
_llm_instance = None
_tokenizer_instance = None

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def _init():
    """Load the model exactly once (called under lock)."""
    global _llm_instance, _tokenizer_instance

    print(f"[llm_loader] Loading {MODEL_NAME} from cache...")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 4-bit quantization (GPU) ----------
    if torch.cuda.is_available():
        # Use bfloat16 if supported (Ampere+ GPUs) for better numerical stability, else float16
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
        except (ImportError, Exception) as e:
            print(f"[llm_loader] 4-bit unavailable ({e}), using {best_dtype}")
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
    _llm_instance = HuggingFacePipeline(pipeline=pipe)
    print(f"[llm_loader] ✅ Model ready — subsequent queries will be fast")


def get_llm():
    """Return the singleton HuggingFacePipeline LLM."""
    global _llm_instance
    if _llm_instance is None:
        with _lock:
            if _llm_instance is None:
                _init()
    return _llm_instance


def get_tokenizer():
    """Return the singleton tokenizer."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        with _lock:
            if _tokenizer_instance is None:
                _init()
    return _tokenizer_instance