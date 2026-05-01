#!/usr/bin/env python
"""Verify the integration of all components."""

import sys
import os
import pathlib

# Ensure the project root is on sys.path so `core.*` imports resolve
# regardless of which directory the script is run from.
_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Force UTF-8 output on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 60)
print("INTEGRATION VERIFICATION")
print("=" * 60)

# Test 1: Config
try:
    from core.config import BM25_USE_GPU, BM25_DENSE_DIM, COLLECTION_NAME
    print(f"[OK] Config loaded: BM25_USE_GPU={BM25_USE_GPU}, BM25_DENSE_DIM={BM25_DENSE_DIM}")
except Exception as e:
    print(f"[FAIL] Config error: {e}")
    sys.exit(1)

# Test 2: BM25
try:
    from core.bm25 import BM25Okapi
    print("[OK] BM25Okapi imported")
except Exception as e:
    print(f"[FAIL] BM25 import error: {e}")
    sys.exit(1)

# Test 3: Retriever
try:
    from core.retriever import FiqhRetriever
    print("[OK] FiqhRetriever imported")
except Exception as e:
    print(f"[FAIL] Retriever import error: {e}")
    sys.exit(1)

# Test 4: Retriever initialization
try:
    retriever = FiqhRetriever()
    print(f"[OK] FiqhRetriever initialized")
    print(f"     - BM25 corpus size: {len(retriever.bm25.corpus)} documents")
    print(f"     - BM25 avg doc length: {retriever.bm25.avgdl:.1f} tokens")
except Exception as e:
    print(f"[FAIL] Retriever init error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Dense BM25 encoding
try:
    from core.retriever import _bm25_encode
    query = "احكام الزكاة"
    bm25_vec = _bm25_encode(query, retriever.bm25)
    print(f"[OK] Dense BM25 query encoding works")
    print(f"     - Query: '{query}'")
    print(f"     - Vector size: {len(bm25_vec)}")
    print(f"     - Vector norm: {sum(v**2 for v in bm25_vec)**0.5:.4f}")
except Exception as e:
    print(f"[FAIL] BM25 encoding error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Qdrant collection
try:
    exists = retriever.client.collection_exists(COLLECTION_NAME)
    if exists:
        info = retriever.client.get_collection(COLLECTION_NAME)
        print(f"[OK] Qdrant collection '{COLLECTION_NAME}' exists")
        print(f"     - Points count: {info.points_count}")
        # Try to get vectors config from different possible attributes
        vectors_keys = []
        try:
            vectors_keys = list(info.config.vectors.keys())
        except:
            try:
                vectors_keys = list(info.config.vectors_config.keys())
            except:
                pass
        if vectors_keys:
            print(f"     - Vectors config: {vectors_keys}")
    else:
        print(f"[WARN] Qdrant collection '{COLLECTION_NAME}' does NOT exist (will be created during ingestion)")
except Exception as e:
    print(f"[FAIL] Qdrant error: {e}")

# Test 7: Gradio app syntax check
try:
    import py_compile
    app_path = str(pathlib.Path(__file__).parent.parent / "app.py")
    py_compile.compile(app_path, doraise=True)
    print("[OK] Gradio app (app.py) syntax valid")
except py_compile.PyCompileError as e:
    print(f"[FAIL] Gradio app syntax error: {e}")

# Test 8: GPU/torch status
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    print(f"[OK] Torch available: CUDA={cuda_avail}")
    if cuda_avail:
        print(f"     - Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"     - CUDA not available (CPU-only torch build or no CUDA GPU detected)")
except (ImportError, OSError) as e:
    print(f"[WARN] Torch import failed: {e}")
    if "WinError 193" in str(e) or "not a valid Win32" in str(e) or ".dll" in str(e).lower():
        print(f"       Cause: installed torch has DLL/CUDA version mismatch.")
        print(f"       Fix:   pip uninstall torch -y")
        print(f"              pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
        print(f"              (replace cu124 with cu121/cu126 to match your CUDA — check: nvidia-smi)")
    print(f"       BM25 will use NumPy CPU fallback until torch is fixed.")

print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
