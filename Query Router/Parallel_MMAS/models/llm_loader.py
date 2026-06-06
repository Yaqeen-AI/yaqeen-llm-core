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
    pass

_hf_home = os.getenv("HF_HOME")
if _hf_home:
    os.environ["HF_HOME"] = _hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

try:
    import torch
except Exception as e:
    print(f"[llm_loader] -> Torch not available ({e}), using dummy stub.")
    class _DummyCuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def is_bf16_supported():
            return False
    class _DummyTorch:
        cuda = _DummyCuda()
        float32 = None
    torch = _DummyTorch()

_lock = threading.Lock()
_llm_instance = None
_tokenizer_instance = None

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def _init():
    """Load the model exactly once (called under lock)."""
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
                def __call__(self, prompt):
                    return "[Dummy LLM] Model unavailable."
                def invoke(self, prompt):
                    return self(prompt)
            _llm_instance = _DummyLLM()

        print(f"[llm_loader] Model ready — subsequent queries will be fast")
    except Exception as exc:
        print(f"[llm_loader] WARNING: Failed to load model ({exc}); using dummy LLM.")
        class _DummyLLM:
            def __call__(self, prompt):
                return "[Dummy LLM] Model unavailable."
            def invoke(self, prompt):
                return self(prompt)
        _llm_instance = _DummyLLM()
        _tokenizer_instance = None


def get_llm():
    """Return the singleton LLM."""
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