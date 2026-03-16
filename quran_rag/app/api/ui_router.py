from __future__ import annotations

from textwrap import dedent

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])


def build_test_ui_html() -> str:
    return dedent(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>YaqeenAI Quran RAG Test UI</title>
          <style>
            :root {
              color-scheme: light;
              --bg: #f6f1e8;
              --bg-accent: #efe4cf;
              --panel: rgba(255, 252, 247, 0.88);
              --panel-strong: #fffaf2;
              --border: rgba(85, 62, 34, 0.14);
              --text: #23170d;
              --muted: #70563a;
              --brand: #8a5a1f;
              --brand-strong: #603b14;
              --ok: #1f6a42;
              --error: #972d2d;
              --shadow: 0 24px 60px rgba(83, 53, 19, 0.12);
              font-family: "IBM Plex Sans Arabic", "Segoe UI", Tahoma, sans-serif;
            }

            * {
              box-sizing: border-box;
            }

            body {
              margin: 0;
              min-height: 100vh;
              color: var(--text);
              background:
                radial-gradient(circle at top left, rgba(218, 175, 107, 0.28), transparent 32%),
                radial-gradient(circle at bottom right, rgba(117, 147, 92, 0.18), transparent 28%),
                linear-gradient(180deg, var(--bg) 0%, #f9f5ee 100%);
            }

            .page {
              width: min(1180px, calc(100% - 32px));
              margin: 0 auto;
              padding: 32px 0 40px;
            }

            .hero {
              display: grid;
              gap: 16px;
              margin-bottom: 24px;
              padding: 28px;
              border: 1px solid var(--border);
              border-radius: 28px;
              background:
                linear-gradient(135deg, rgba(255, 250, 242, 0.96), rgba(243, 231, 206, 0.82)),
                var(--panel);
              box-shadow: var(--shadow);
            }

            .eyebrow {
              margin: 0;
              font-size: 0.84rem;
              letter-spacing: 0.12em;
              text-transform: uppercase;
              color: var(--brand);
              font-weight: 700;
            }

            h1 {
              margin: 0;
              font-size: clamp(2rem, 5vw, 3.4rem);
              line-height: 1.05;
              font-family: Georgia, "Times New Roman", serif;
            }

            .hero p {
              margin: 0;
              max-width: 760px;
              color: var(--muted);
              font-size: 1rem;
              line-height: 1.65;
            }

            .status-bar {
              display: flex;
              flex-wrap: wrap;
              gap: 12px;
              align-items: center;
              margin-top: 8px;
            }

            .status-chip {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              padding: 10px 14px;
              border-radius: 999px;
              border: 1px solid var(--border);
              background: rgba(255, 255, 255, 0.72);
              font-size: 0.95rem;
            }

            .status-chip::before {
              content: "";
              width: 10px;
              height: 10px;
              border-radius: 50%;
              background: #c29f6c;
            }

            .status-chip.healthy::before {
              background: var(--ok);
            }

            .status-chip.error::before {
              background: var(--error);
            }

            .grid {
              display: grid;
              gap: 20px;
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .panel {
              display: grid;
              gap: 16px;
              align-content: start;
              padding: 22px;
              border: 1px solid var(--border);
              border-radius: 24px;
              background: var(--panel);
              backdrop-filter: blur(12px);
              box-shadow: var(--shadow);
            }

            .panel h2 {
              margin: 0;
              font-size: 1.35rem;
              font-family: Georgia, "Times New Roman", serif;
            }

            .panel p {
              margin: 0;
              color: var(--muted);
              line-height: 1.55;
            }

            form {
              display: grid;
              gap: 14px;
            }

            label {
              display: grid;
              gap: 8px;
              font-weight: 600;
              font-size: 0.96rem;
            }

            input,
            select,
            button,
            textarea {
              width: 100%;
              border-radius: 14px;
              border: 1px solid rgba(101, 76, 46, 0.18);
              background: var(--panel-strong);
              color: var(--text);
              font: inherit;
            }

            input,
            select,
            textarea {
              padding: 13px 14px;
            }

            input:focus,
            select:focus,
            textarea:focus {
              outline: 2px solid rgba(138, 90, 31, 0.2);
              border-color: rgba(138, 90, 31, 0.42);
            }

            .controls {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 12px;
            }

            .toggles {
              display: flex;
              gap: 12px;
              flex-wrap: wrap;
            }

            .toggle {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              padding: 10px 12px;
              border-radius: 14px;
              border: 1px solid var(--border);
              background: rgba(255, 255, 255, 0.65);
              font-weight: 500;
            }

            .toggle input {
              width: auto;
              margin: 0;
            }

            button {
              cursor: pointer;
              padding: 14px 16px;
              border: none;
              color: #fff8f0;
              background: linear-gradient(135deg, var(--brand) 0%, var(--brand-strong) 100%);
              font-weight: 700;
              letter-spacing: 0.01em;
              transition: transform 120ms ease, filter 120ms ease;
            }

            button:hover {
              transform: translateY(-1px);
              filter: brightness(1.03);
            }

            button:disabled {
              cursor: progress;
              opacity: 0.7;
              transform: none;
            }

            .results {
              display: grid;
              gap: 12px;
              min-height: 280px;
            }

            .placeholder,
            .message,
            .answer-box,
            .result-card,
            details {
              border: 1px solid var(--border);
              border-radius: 18px;
              background: rgba(255, 255, 255, 0.7);
            }

            .placeholder,
            .message,
            .answer-box {
              padding: 16px;
            }

            .placeholder {
              color: var(--muted);
            }

            .message.error {
              color: var(--error);
              background: rgba(255, 240, 240, 0.9);
            }

            .message.loading {
              color: var(--brand-strong);
            }

            .result-card {
              padding: 16px;
              display: grid;
              gap: 10px;
            }

            .result-card header,
            .answer-meta {
              display: flex;
              justify-content: space-between;
              gap: 12px;
              align-items: flex-start;
              flex-wrap: wrap;
            }

            .badge-row {
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
            }

            .badge {
              display: inline-flex;
              align-items: center;
              gap: 6px;
              padding: 6px 10px;
              border-radius: 999px;
              background: rgba(138, 90, 31, 0.09);
              color: var(--brand-strong);
              font-size: 0.84rem;
              font-weight: 700;
            }

            .result-text,
            .answer-text,
            pre {
              margin: 0;
              white-space: pre-wrap;
              line-height: 1.7;
              word-break: break-word;
            }

            .diagnostics {
              display: grid;
              gap: 8px;
            }

            details {
              overflow: hidden;
            }

            summary {
              cursor: pointer;
              padding: 14px 16px;
              font-weight: 700;
            }

            pre {
              padding: 0 16px 16px;
              font-size: 0.85rem;
              color: #3d2c1b;
              overflow-x: auto;
            }

            .list {
              display: grid;
              gap: 10px;
            }

            .citation {
              padding: 12px 14px;
              border-radius: 14px;
              border: 1px solid var(--border);
              background: rgba(247, 242, 232, 0.96);
            }

            .small {
              font-size: 0.88rem;
              color: var(--muted);
            }

            @media (max-width: 900px) {
              .grid {
                grid-template-columns: 1fr;
              }
            }

            @media (max-width: 640px) {
              .page {
                width: min(100% - 20px, 1180px);
                padding-top: 20px;
              }

              .hero,
              .panel {
                padding: 18px;
                border-radius: 20px;
              }

              .controls {
                grid-template-columns: 1fr;
              }
            }
          </style>
        </head>
        <body>
          <main class="page">
            <section class="hero">
              <p class="eyebrow">Quran RAG Test Bench</p>
              <h1>Retriever and generation UI for manual QA.</h1>
              <p>
                Use the left panel to inspect pure retrieval output. Use the right panel to test
                the full retriever plus answer-generation pipeline with citations and retrieval trace.
              </p>
              <div class="status-bar">
                <div id="health-status" class="status-chip">Checking backend health...</div>
                <div id="generation-status" class="status-chip">Checking generation service...</div>
              </div>
            </section>

            <section class="grid">
              <article class="panel">
                <h2>Retriever Only</h2>
                <p>Calls <code>POST /api/v1/retrieve</code> and shows ranking, metadata, and pipeline diagnostics.</p>
                <form id="retriever-form">
                  <label for="retriever-query">
                    Retriever query
                    <input
                      id="retriever-query"
                      name="retriever-query"
                      type="text"
                      placeholder="Example: آية الكرسي or patience in the Quran"
                      required
                    />
                  </label>

                  <div class="controls">
                    <label for="retriever-top-k">
                      Top K
                      <input id="retriever-top-k" type="number" min="1" max="20" value="5" />
                    </label>

                    <label for="retriever-language">
                      Language filter
                      <select id="retriever-language">
                        <option value="">Auto</option>
                        <option value="ar">Arabic</option>
                        <option value="en">English</option>
                      </select>
                    </label>
                  </div>

                  <div class="toggles">
                    <label class="toggle" for="retriever-hybrid">
                      <input id="retriever-hybrid" type="checkbox" checked />
                      Use hybrid
                    </label>
                    <label class="toggle" for="retriever-rerank">
                      <input id="retriever-rerank" type="checkbox" checked />
                      Use reranking
                    </label>
                  </div>

                  <button id="retriever-submit" type="submit">Run retrieval</button>
                </form>

                <div id="retriever-results" class="results">
                  <div class="placeholder">Retriever results will appear here.</div>
                </div>
              </article>

              <article class="panel">
                <h2>Full Project With Generation</h2>
                <p>Calls <code>POST /api/v1/answer</code> and shows the generated answer, citations, and retrieval trace.</p>
                <form id="answer-form">
                  <label for="answer-query">
                    Generation query
                    <input
                      id="answer-query"
                      name="answer-query"
                      type="text"
                      placeholder="Example: ما مضمون آية الكرسي؟"
                      required
                    />
                  </label>

                  <div class="controls">
                    <label for="answer-top-k">
                      Top K
                      <input id="answer-top-k" type="number" min="1" max="10" value="4" />
                    </label>

                    <label for="answer-context-window">
                      Context window
                      <input id="answer-context-window" type="number" min="0" max="5" value="1" />
                    </label>
                  </div>

                  <div class="toggles">
                    <label class="toggle" for="answer-hybrid">
                      <input id="answer-hybrid" type="checkbox" checked />
                      Use hybrid
                    </label>
                    <label class="toggle" for="answer-rerank">
                      <input id="answer-rerank" type="checkbox" checked />
                      Use reranking
                    </label>
                  </div>

                  <button id="answer-submit" type="submit">Run full RAG</button>
                </form>

                <div id="answer-results" class="results">
                  <div class="placeholder">Generated answers will appear here.</div>
                </div>
              </article>
            </section>
          </main>

          <script>
            const healthStatus = document.getElementById("health-status");
            const generationStatus = document.getElementById("generation-status");
            const retrieverResults = document.getElementById("retriever-results");
            const answerResults = document.getElementById("answer-results");
            const retrieverForm = document.getElementById("retriever-form");
            const answerForm = document.getElementById("answer-form");
            const retrieverSubmit = document.getElementById("retriever-submit");
            const answerSubmit = document.getElementById("answer-submit");

            function prettyJson(payload) {
              return JSON.stringify(payload, null, 2);
            }

            function escapeHtml(value) {
              return String(value)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
            }

            function renderMessage(container, text, kind) {
              container.innerHTML = `<div class="message ${kind}">${escapeHtml(text)}</div>`;
            }

            function renderRetriever(payload) {
              const resultsHtml = payload.results.length
                ? payload.results.map((result, index) => {
                    const meta = result.metadata || {};
                    const ref = meta.ayah_ref || [meta.surah_number, meta.ayah_number_in_surah].filter(Boolean).join(":") || "n/a";
                    const badges = [
                      `#${index + 1}`,
                      result.retrieval_method || "unknown",
                      `score ${Number(result.score || 0).toFixed(4)}`,
                      ref,
                      meta.content_type || "n/a"
                    ].map((item) => `<span class="badge">${escapeHtml(item)}</span>`).join("");

                    return `
                      <article class="result-card">
                        <header>
                          <div class="badge-row">${badges}</div>
                          <div class="small">${escapeHtml(meta.edition_name || meta.edition_identifier || "")}</div>
                        </header>
                        <p class="result-text">${escapeHtml(result.text || "")}</p>
                        <div class="small">
                          Surah: ${escapeHtml(meta.surah_number ?? "n/a")} |
                          Juz: ${escapeHtml(meta.juz ?? "n/a")} |
                          Language: ${escapeHtml(meta.language || "n/a")}
                        </div>
                      </article>
                    `;
                  }).join("")
                : `<div class="message">No retrieval results returned.</div>`;

              const diagnostics = `
                <div class="result-card diagnostics">
                  <div class="badge-row">
                    <span class="badge">Latency ${escapeHtml(payload.latency_ms ?? 0)} ms</span>
                    <span class="badge">Semantic ${escapeHtml(payload.total_candidates_semantic ?? 0)}</span>
                    <span class="badge">BM25 ${escapeHtml(payload.total_candidates_bm25 ?? 0)}</span>
                    <span class="badge">Fusion ${escapeHtml(payload.total_after_fusion ?? 0)}</span>
                    <span class="badge">Rerank ${escapeHtml(payload.total_after_reranking ?? 0)}</span>
                  </div>
                  <div class="small">${escapeHtml((payload.pipeline_steps || []).join(" -> "))}</div>
                </div>
              `;

              retrieverResults.innerHTML = `
                ${diagnostics}
                ${resultsHtml}
                <details>
                  <summary>Raw JSON</summary>
                  <pre>${escapeHtml(prettyJson(payload))}</pre>
                </details>
              `;
            }

            function renderAnswer(payload) {
              const citations = payload.citations && payload.citations.length
                ? payload.citations.map((citation) => `
                    <div class="citation">
                      <div class="badge-row">
                        <span class="badge">${escapeHtml(citation.ayah_ref || "n/a")}</span>
                        <span class="badge">${escapeHtml(citation.content_type || "n/a")}</span>
                        <span class="badge">score ${escapeHtml(Number(citation.score || 0).toFixed(4))}</span>
                      </div>
                      <p class="result-text">${escapeHtml(citation.text || "")}</p>
                    </div>
                  `).join("")
                : `<div class="message">No citations were returned.</div>`;

              const retrieval = payload.retrieval || {};

              answerResults.innerHTML = `
                <section class="answer-box">
                  <div class="answer-meta">
                    <div class="badge-row">
                      <span class="badge">${escapeHtml(payload.model_name || "unknown model")}</span>
                      <span class="badge">Latency ${escapeHtml(retrieval.latency_ms ?? 0)} ms</span>
                      <span class="badge">Results ${escapeHtml((retrieval.results || []).length)}</span>
                    </div>
                  </div>
                  <p class="answer-text">${escapeHtml(payload.answer || "")}</p>
                </section>

                <section class="list">
                  ${citations}
                </section>

                <div class="result-card diagnostics">
                  <div class="small">${escapeHtml((retrieval.pipeline_steps || []).join(" -> "))}</div>
                </div>

                <details>
                  <summary>Raw JSON</summary>
                  <pre>${escapeHtml(prettyJson(payload))}</pre>
                </details>
              `;
            }

            async function requestJson(url, payload) {
              const response = await fetch(url, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
              });

              let data = null;
              try {
                data = await response.json();
              } catch (_error) {
                data = null;
              }

              if (!response.ok) {
                const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
                throw new Error(detail);
              }

              return data;
            }

            async function refreshHealth() {
              try {
                const response = await fetch("/api/v1/health");
                const data = await response.json();

                healthStatus.textContent = `Backend healthy | BM25 ${data.bm25_corpus_size} docs`;
                healthStatus.className = "status-chip healthy";

                if (data.generation_ready) {
                  generationStatus.textContent = "Generation ready";
                  generationStatus.className = "status-chip healthy";
                } else {
                  generationStatus.textContent = "Generation unavailable";
                  generationStatus.className = "status-chip error";
                }
              } catch (_error) {
                healthStatus.textContent = "Backend health check failed";
                healthStatus.className = "status-chip error";
                generationStatus.textContent = "Generation status unavailable";
                generationStatus.className = "status-chip error";
              }
            }

            retrieverForm.addEventListener("submit", async (event) => {
              event.preventDefault();
              retrieverSubmit.disabled = true;
              renderMessage(retrieverResults, "Running retrieval...", "loading");

              const payload = {
                query: document.getElementById("retriever-query").value.trim(),
                top_k: Number(document.getElementById("retriever-top-k").value || 5),
                use_hybrid: document.getElementById("retriever-hybrid").checked,
                use_reranking: document.getElementById("retriever-rerank").checked
              };

              const language = document.getElementById("retriever-language").value;
              if (language) {
                payload.language = language;
              }

              try {
                const data = await requestJson("/api/v1/retrieve", payload);
                renderRetriever(data);
              } catch (error) {
                renderMessage(retrieverResults, error.message, "error");
              } finally {
                retrieverSubmit.disabled = false;
              }
            });

            answerForm.addEventListener("submit", async (event) => {
              event.preventDefault();
              answerSubmit.disabled = true;
              renderMessage(answerResults, "Running full RAG generation...", "loading");

              const payload = {
                query: document.getElementById("answer-query").value.trim(),
                top_k: Number(document.getElementById("answer-top-k").value || 4),
                context_window: Number(document.getElementById("answer-context-window").value || 1),
                use_hybrid: document.getElementById("answer-hybrid").checked,
                use_reranking: document.getElementById("answer-rerank").checked
              };

              try {
                const data = await requestJson("/api/v1/answer", payload);
                renderAnswer(data);
              } catch (error) {
                renderMessage(answerResults, error.message, "error");
              } finally {
                answerSubmit.disabled = false;
              }
            });

            refreshHealth();
          </script>
        </body>
        </html>
        """
    ).strip()


@router.get("/ui", include_in_schema=False, response_class=HTMLResponse)
async def test_ui() -> HTMLResponse:
    return HTMLResponse(content=build_test_ui_html())
