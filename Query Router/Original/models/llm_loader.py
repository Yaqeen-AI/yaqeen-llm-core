"""
Singleton LLM loader with 4-bit quantization.

Provides get_llm() and get_tokenizer() — thread-safe, lazy-initialized.
The model is loaded once on first access and reused across all modules.
"""

import os
import warnings
import threading

# ── Set HF_HOME *before* importing transformers / huggingface_hub ──
try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_ENV_PATH)
except ImportError:
    # dotenv not installed; skip loading .env
    pass

_hf_home = os.getenv("HF_HOME")
if _hf_home:
    os.environ["HF_HOME"] = _hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")

# Suppress noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Optional import of torch – if unavailable we fall back to a dummy stub
try:
    import torch
except Exception as e:
    print(f"[llm_loader] -> Torch not available ({e}), using dummy torch stub.")
    class _DummyCuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def is_bf16_supported():
            return False
    class _DummyTorch:
        cuda = _DummyCuda()
    torch = _DummyTorch()

# Transformers and huggingface_hub are imported lazily inside _init

_lock = threading.Lock()
_llm_instance = None
_tokenizer_instance = None

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def _init():
    """Load the model exactly once (called under lock).
    If any import or loading step fails (e.g., missing torch or CUDA DLLs),
    we fall back to a lightweight dummy LLM that simply echoes the question.
    """
    global _llm_instance, _tokenizer_instance
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from huggingface_hub import login
        print(f"[llm_loader] Loading {MODEL_NAME} from cache...")
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(hf_token, add_to_git_credential=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Attempt 4‑bit quantization only if real torch is available
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
                print(f"[llm_loader] 4‑bit load failed ({e}), falling back to full‑precision CPU model.")
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
            print(f"[llm_loader] LangChain HuggingFacePipeline import failed ({e}), using dummy LLM.")
            class _DummyLLM:
                def __call__(self, prompt):
                    return "[Dummy LLM] I'm currently running without a real language model. Please install torch and transformers for full functionality."
                def invoke(self, prompt):
                    return self(prompt)
            _llm_instance = _DummyLLM()
        print(f"[llm_loader] Model ready -- subsequent queries will be fast")
    except Exception as exc:
        # Any failure - dummy LLM and simple whitespace tokenizer fallback
        print(f"[llm_loader] WARNING: Failed to load real model ({exc}); using dummy LLM.")
        class _DummyLLM:
            def __call__(self, prompt):
                return "[Dummy LLM] I'm currently running without a real language model. Please install torch and transformers for full functionality."
            def invoke(self, prompt):
                return self(prompt)
        _llm_instance = _DummyLLM()
        _tokenizer_instance = None


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