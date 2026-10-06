"""
Converts hiring_test.ipynb into a highly polished academic PDF (hiring_test.pdf) using Playwright.

Features:
- Title page for Computational Biomedicine Lab hiring test
- Verbatim Question 1, Question 2, Question 3 preservation
- Renders Markdown (via Marked.js in browser), Math LaTeX (via KaTeX/MathJax), Python code, outputs, tables, and images
- Prevents code line clipping (word-wrap)
- Prevents table/figure awkward pagination (page-break-inside: avoid)
- Adds running headers and page numbers
- Evaluates completely in Playwright Headless Chromium
"""

import json, os, sys, base64

def generate_pdf():
    print("Reading hiring_test.ipynb...")
    with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells_json = json.dumps(nb['cells'], ensure_ascii=False)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Computational Biomedicine Lab — Technical Test</title>

<!-- Marked.js for Markdown parsing -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<!-- KaTeX for math rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

<!-- Highlight.js for code syntax highlighting -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.8.0/styles/github.min.css">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.8.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.8.0/languages/python.min.js"></script>

<style>
  @page {{
    size: A4;
    margin: 18mm 15mm 18mm 15mm;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.45;
    color: #1a1a1a;
    background: #ffffff;
    margin: 0;
    padding: 0;
  }}

  /* Title Page */
  .title-page {{
    height: 90vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    page-break-after: always;
    border-bottom: 2px solid #003366;
    padding: 40px 20px;
  }}

  .title-header {{
    font-size: 24pt;
    font-weight: 700;
    color: #003366;
    margin-bottom: 15px;
    letter-spacing: -0.5px;
  }}

  .title-subtitle {{
    font-size: 14pt;
    font-weight: 500;
    color: #444444;
    margin-bottom: 40px;
  }}

  .title-divider {{
    width: 100px;
    height: 3px;
    background-color: #003366;
    margin: 20px auto 40px auto;
  }}

  .title-meta {{
    font-size: 11pt;
    color: #222222;
    line-height: 1.8;
  }}

  .title-meta strong {{
    color: #003366;
  }}

  /* Cell containers */
  .cell {{
    margin-bottom: 12px;
    page-break-inside: auto;
  }}

  .cell-markdown {{
    margin-top: 10px;
    margin-bottom: 10px;
  }}

  .cell-code {{
    margin-top: 8px;
    margin-bottom: 12px;
    page-break-inside: avoid;
  }}

  /* Markdown Typography */
  h1 {{
    font-size: 16pt;
    color: #003366;
    border-bottom: 1.5px solid #003366;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}

  h2 {{
    font-size: 13pt;
    color: #1a365d;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 11pt;
    color: #2b6cb0;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 4px;
    margin-bottom: 8px;
  }}

  ul, ol {{
    margin-top: 4px;
    margin-bottom: 8px;
    padding-left: 20px;
  }}

  li {{
    margin-bottom: 3px;
  }}

  blockquote {{
    border-left: 4px solid #3182ce;
    background-color: #ebf8ff;
    margin: 8px 0;
    padding: 6px 12px;
    color: #2c5282;
    font-style: normal;

  }}

  /* Code Cell Input */
  .code-header {{
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 7.5pt;
    color: #718096;
    background: #edf2f7;
    padding: 3px 8px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #cbd5e0;
    border-bottom: none;
    display: inline-block;
    margin-bottom: 0;
  }}

  pre.code-input {{
    background-color: #f7fafc;
    border: 1px solid #cbd5e0;
    border-radius: 0 4px 4px 4px;
    padding: 8px 10px;
    margin: 0 0 6px 0;
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 8pt;
    line-height: 1.35;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: hidden;
  }}

  /* Code Cell Output */
  .output-container {{
    margin-top: 4px;
    margin-left: 8px;
    margin-bottom: 8px;
  }}

  pre.output-stream {{
    background-color: #ffffff;
    border-left: 3px solid #4a5568;
    padding: 6px 10px;
    margin: 4px 0;
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 7.8pt;
    line-height: 1.3;
    color: #2d3748;
    white-space: pre-wrap;
    word-break: break-word;
  }}

  .output-image {{
    text-align: center;
    margin: 10px 0;
    page-break-inside: avoid;
  }}

  .output-image img {{
    max-width: 92%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}

  /* Tables */
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}

  th, td {{
    border: 1px solid #cbd5e0;
    padding: 5px 8px;
    text-align: left;
  }}

  th {{
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 600;
  }}

  tr:nth-child(even) {{
    background-color: #f7fafc;
  }}

  /* Question Boxes */
  .question-box {{
    background-color: #f0f4f8;
    border-left: 5px solid #003366;
    padding: 12px 16px;
    margin: 15px 0;
    border-radius: 0 4px 4px 0;
  }}

  /* Code line wrap */
  code {{
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    background-color: #edf2f7;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.5pt;
  }}
</style>
</head>
<body>

<div class="title-page">
  <div class="title-header">Computational Biomedicine Lab</div>
  <div class="title-subtitle">Research Position / PhD & Postdoc Technical Test</div>
  <div class="title-divider"></div>
  <div class="title-meta">
    <p><strong>Candidate:</strong> Jose Roy</p>
    <p><strong>Dataset:</strong> UCI Breast Cancer Wisconsin (Diagnostic)</p>
    <p><strong>Task:</strong> Technical Hiring Test Evaluation (Questions 1–7)</p>
    <p><strong>Date:</strong> October 2026</p>
  </div>
</div>

<div id="notebook-container"></div>

<script>
  const cells = {cells_json};

  marked.setOptions({{
    gfm: true,
    breaks: true
  }});

  const container = document.getElementById('notebook-container');

  cells.forEach((cell, idx) => {{
    const cellDiv = document.createElement('div');
    cellDiv.className = 'cell cell-' + cell.cell_type;

    if (cell.cell_type === 'markdown') {{
      const rawText = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
      const htmlContent = marked.parse(rawText);
      cellDiv.innerHTML = htmlContent;
    }} else if (cell.cell_type === 'code') {{
      const codeSrc = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
      
      // Header tag for code cell execution count
      const execCount = cell.execution_count !== null ? `In [${{cell.execution_count}}]` : 'In [ ]';
      
      let html = `<div class="code-header">${{execCount}}</div>`;
      html += `<pre class="code-input"><code class="language-python">${{escapeHtml(codeSrc)}}</code></pre>`;

      // Render outputs
      if (cell.outputs && cell.outputs.length > 0) {{
        html += `<div class="output-container">`;
        cell.outputs.forEach(out => {{
          if (out.output_type === 'stream') {{
            const streamText = Array.isArray(out.text) ? out.text.join('') : out.text;
            html += `<pre class="output-stream">${{escapeHtml(streamText)}}</pre>`;
          }} else if (out.output_type === 'execute_result' || out.output_type === 'display_data') {{
            if (out.data) {{
              if (out.data['image/png']) {{
                html += `<div class="output-image"><img src="data:image/png;base64,${{out.data['image/png']}}" alt="Plot"></div>`;
              }} else if (out.data['text/plain']) {{
                const plainText = Array.isArray(out.data['text/plain']) ? out.data['text/plain'].join('') : out.data['text/plain'];
                html += `<pre class="output-stream">${{escapeHtml(plainText)}}</pre>`;
              }}
            }}
          }}
        }});
        html += `</div>`;
      }}
      cellDiv.innerHTML = html;
    }}

    container.appendChild(cellDiv);
  }});

  function escapeHtml(text) {{
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }}

  // Trigger syntax highlighting
  document.querySelectorAll('pre code').forEach((block) => {{
    hljs.highlightElement(block);
  }});

  // Trigger KaTeX math rendering
  renderMathInElement(document.body, {{
    delimiters: [
      {{left: "$$", right: "$$", display: true}},
      {{left: "$", right: "$", display: false}},
      {{left: "\\(", right: "\\)", display: false}},
      {{left: "\\[", right: "\\]", display: true}}
    ],
    throwOnError: false
  }});
</script>

</body>
</html>
"""

    with open('hiring_test_render.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

    print("Generated hiring_test_render.html")
    print("Launching Playwright to render hiring_test.pdf...")

    from playwright.sync_api import sync_playwright

    html_abs_path = "file:///" + os.path.abspath('hiring_test_render.html').replace('\\', '/')

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_abs_path, wait_until='networkidle')

        # Give math rendering and fonts a brief moment
        page.wait_for_timeout(2000)

        # PDF export settings
        page.pdf(
            path='hiring_test.pdf',
            format='A4',
            print_background=True,
            margin={
                'top': '18mm',
                'bottom': '18mm',
                'left': '15mm',
                'right': '15mm'
            },
            display_header_footer=True,
            header_template='''
                <div style="font-size:8pt; font-family:sans-serif; color:#718096; width:100%; text-align:center; padding-top:4mm;">
                    Computational Biomedicine Lab — Technical Test Evaluation | Candidate: Jose Roy
                </div>
            ''',
            footer_template='''
                <div style="font-size:8pt; font-family:sans-serif; color:#718096; width:100%; text-align:center; padding-bottom:4mm;">
                    Page <span class="pageNumber"></span> of <span class="totalPages"></span>
                </div>
            '''
        )
        browser.close()

    print(f"\nSuccessfully generated hiring_test.pdf! ({os.path.getsize('hiring_test.pdf')//1024} KB)")

if __name__ == '__main__':
    generate_pdf()
