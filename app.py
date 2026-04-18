"""
FiqhRAG — Arabic web UI (Gradio 6)
    python app.py
"""

import gradio as gr
from core.retriever import FiqhRetriever
from core.generator import generate_answer
from core.cache import TwoTierCache

retriever = FiqhRetriever()
cache     = TwoTierCache()


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def ask(query: str, history: list):
    if not query.strip():
        yield history, "", gr.update(visible=False)
        return

    # ── Cache check ───────────────────────────────────────────────────────
    cached_answer = cache.get(query)
    if cached_answer is not None:
        full_response = (
            f"{cached_answer}\n\n"
            "<details>\n<summary>⚡ من الذاكرة المؤقتة</summary>\n\n"
            "*تمت الإجابة من الذاكرة المؤقتة — لم تُستدعَ نماذج التوليد.*\n</details>"
        )
        yield history + [_msg("user", query), _msg("assistant", full_response)], "", gr.update(visible=False)
        return

    yield history + [_msg("user", query)], "", gr.update(visible=True)

    # ── Retrieve ──────────────────────────────────────────────────────────
    try:
        results = retriever.retrieve(query)
    except Exception as e:
        yield history + [_msg("user", query), _msg("assistant", f"⚠️ **خطأ في الاسترجاع:**\n{e}")], "", gr.update(visible=False)
        return

    if not results:
        yield history + [_msg("user", query), _msg("assistant", "لم يُعثر على نتائج ذات صلة في الموسوعة الفقهية الكويتية.")], "", gr.update(visible=False)
        return

    # ── Generate ──────────────────────────────────────────────────────────
    try:
        answer = generate_answer(query, results)
    except Exception as e:
        yield history + [_msg("user", query), _msg("assistant", f"⚠️ **خطأ في التوليد** — هل LM Studio يعمل؟\n{e}")], "", gr.update(visible=False)
        return

    # ── Format sources ────────────────────────────────────────────────────
    sources_lines = []
    for i, r in enumerate(results, 1):
        mazhab_badge = f" · `{r.mazhab_tag()}`" if r.mazhabs else ""
        sources_lines.append(
            f"**[{i}]** {r.short_ref()}{mazhab_badge} · `{r.rerank_score:.3f}`\n"
            f"> {r.chunk_text[:200]}{'…' if len(r.chunk_text) > 200 else ''}"
        )

    full_response = (
        f"{answer}\n\n"
        f"<details>\n<summary>📚 المصادر ({len(results)} مقاطع)</summary>\n\n"
        + "\n\n".join(sources_lines)
        + "\n</details>"
    )

    cache.set(query, answer)
    yield history + [_msg("user", query), _msg("assistant", full_response)], "", gr.update(visible=False)


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Tokens ─────────────────────────────────────── */
:root {
    --navy-950: #0f1628;
    --navy-800: #1a2744;
    --navy-700: #243366;
    --navy-500: #2d4799;
    --navy-100: #e8edf8;
    --gold-500: #c9a84c;
    --gold-300: #e2c97e;
    --gold-100: #fdf6e3;
    --surface:  #ffffff;
    --bg:       #f5f6fa;
    --border:   #dde1ed;
    --text:     #1a1a2e;
    --muted:    #6b7280;
    --r:        12px;
    --shadow:   0 2px 16px rgba(15,22,40,.08);
}

/* ── Page ────────────────────────────────────────── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container { max-width: 900px !important; margin: 0 auto !important; padding: 20px !important; }
footer { display: none !important; }

/* ── Header card ─────────────────────────────────── */
#header-card {
    background: linear-gradient(135deg, var(--navy-950) 0%, var(--navy-800) 55%, var(--navy-700) 100%);
    border-radius: var(--r);
    padding: 30px 32px 24px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 6px 40px rgba(15,22,40,.4);
    position: relative;
    overflow: hidden;
    border-bottom: 3px solid var(--gold-500);
}
#header-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at top, rgba(201,168,76,.08) 0%, transparent 65%);
    pointer-events: none;
}
#title-ar {
    font-family: 'Noto Naskh Arabic', serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 0 4px !important;
    direction: rtl;
    text-shadow: 0 2px 12px rgba(0,0,0,.4);
}
#title-en {
    font-size: .78rem !important;
    color: var(--gold-300) !important;
    letter-spacing: .1em;
    margin: 0 0 18px !important;
    text-transform: uppercase;
    font-weight: 500;
}
#badges {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 7px;
}
.badge {
    background: rgba(201,168,76,.12);
    border: 1px solid rgba(201,168,76,.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .69rem;
    color: var(--gold-300);
    letter-spacing: .03em;
}

/* ── Chat area ───────────────────────────────────── */
#chatbox {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow) !important;
}

/* ── Input panel ─────────────────────────────────── */
#input-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 14px 16px 12px;
    box-shadow: var(--shadow);
    margin-top: 12px;
}
#query-box textarea {
    font-family: 'Noto Naskh Arabic', 'Segoe UI', sans-serif !important;
    font-size: 1.05rem !important;
    direction: rtl !important;
    border: none !important;
    background: transparent !important;
    resize: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
    color: var(--text) !important;
}
#query-box textarea:focus { outline: none !important; box-shadow: none !important; }
#query-box textarea::placeholder { color: var(--muted) !important; }

/* ── Buttons ─────────────────────────────────────── */
#send-btn {
    background: linear-gradient(135deg, var(--gold-500), #a8893a) !important;
    color: var(--navy-950) !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: .88rem !important;
    padding: 10px 20px !important;
    box-shadow: 0 2px 12px rgba(201,168,76,.45) !important;
    transition: opacity .15s, transform .1s !important;
    white-space: nowrap;
}
#send-btn:hover  { opacity: .88 !important; transform: translateY(-1px) !important; }
#send-btn:active { transform: translateY(0) !important; }

#clear-btn {
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: transparent !important;
    font-size: .8rem !important;
    transition: border-color .15s, color .15s !important;
}
#clear-btn:hover { border-color: #ef4444 !important; color: #ef4444 !important; }

/* ── Loading bar ─────────────────────────────────── */
#loading-bar {
    text-align: center;
    color: var(--gold-500);
    font-size: .84rem;
    padding: 8px 0 4px;
    font-style: italic;
}

/* ── Example pills ───────────────────────────────── */
.examples-holder { margin-top: 8px !important; }
.example-label   { display: none !important; }
.example {
    font-family: 'Noto Naskh Arabic', sans-serif !important;
    font-size: .9rem !important;
    direction: rtl !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    transition: border-color .15s, background .15s !important;
    padding: 6px 12px !important;
}
.example:hover {
    border-color: var(--gold-500) !important;
    background: var(--gold-100) !important;
}

/* ── Stats strip ─────────────────────────────────── */
#stats-strip {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 6px 20px;
    background: var(--navy-950);
    border: 1px solid var(--navy-800);
    border-radius: var(--r);
    padding: 10px 16px;
    margin-top: 10px;
    font-size: .74rem;
    color: rgba(255,255,255,.5);
}
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────

HEADER_HTML = """
<div id="header-card">
  <p id="title-ar">الموسوعة الفقهية الكويتية — بحث ذكي</p>
  <p id="title-en">FiqhRAG · Islamic Jurisprudence AI Search</p>
  <div id="badges">
    <span class="badge">⚡ Jina Embeddings v3</span>
    <span class="badge">🔍 Hybrid TF-IDF + Semantic</span>
    <span class="badge">🏆 Jina Reranker v2</span>
    <span class="badge">🤖 Gemma 4 · LM Studio</span>
    <span class="badge">🗄️ Two-Tier Cache</span>
    <span class="badge">📚 46 مجلداً</span>
  </div>
</div>
"""

STATS_HTML = """
<div id="stats-strip">
  <span>🗄️ Qdrant · Local Vector DB</span>
  <span>📖 16,971 مقطع مفهرس</span>
  <span>🔎 Top-50 → Rerank → Top-10</span>
  <span>🌐 NFKC · Char n-grams (3–5)</span>
  <span>🕌 Mazhab-aware · Two-Tier Cache</span>
</div>
"""

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="الموسوعة الفقهية — بحث ذكي") as demo:

    gr.HTML(HEADER_HTML)

    chatbot = gr.Chatbot(
        value=[],
        label="",
        elem_id="chatbox",
        height=520,
        rtl=True,
        render_markdown=True,
        layout="bubble",
        avatar_images=(None, "https://img.icons8.com/color/48/quran.png"),
    )

    loading = gr.Markdown("جارٍ البحث والتوليد...", visible=False, elem_id="loading-bar")

    with gr.Group(elem_id="input-panel"):
        with gr.Row():
            query_box = gr.Textbox(
                placeholder="اكتب سؤالك الفقهي هنا...  ✦  اضغط Enter للإرسال",
                show_label=False,
                scale=9,
                lines=1,
                elem_id="query-box",
                rtl=True,
                autofocus=True,
                container=False,
            )
            send_btn = gr.Button("إرسال ↵", scale=1, elem_id="send-btn", variant="primary", min_width=90)

    with gr.Row():
        clear_btn = gr.Button("🗑  مسح المحادثة", size="sm", elem_id="clear-btn")

    gr.Examples(
        examples=[
            ["ما حكم الصلاة في الأرض المغصوبة؟"],
            ["ما شروط صحة عقد البيع عند المذاهب الأربعة؟"],
            ["ما حكم الزكاة على الذهب والفضة؟"],
            ["ما حكم الطلاق في حالة الغضب الشديد؟"],
            ["ما حكم صوم من أفطر ناسياً في رمضان؟"],
            ["ما حكم الوضوء بالماء المستعمل؟"],
        ],
        inputs=query_box,
        label="أمثلة — انقر لتحديد",
    )

    gr.HTML(STATS_HTML)

    send_btn.click(ask, [query_box, chatbot], [chatbot, query_box, loading])
    query_box.submit(ask, [query_box, chatbot], [chatbot, query_box, loading])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, query_box])


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        css=CSS,
    )
