#!/usr/bin/env python3
"""Chuyển một file Markdown -> PDF qua Chrome headless (hỗ trợ tiếng Việt)."""
import sys, subprocess, tempfile, os, pathlib
import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Helvetica Neue", "Segoe UI", Arial, sans-serif;
       font-size: 11.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4px; }
h2 { font-size: 14pt; margin: 18px 0 6px; border-bottom: 2px solid #e2e2e2; padding-bottom: 3px; }
h3 { font-size: 12pt; margin: 12px 0 4px; }
p, li { margin: 4px 0; }
ul { margin: 4px 0 8px 0; padding-left: 20px; }
strong { color: #000; }
hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10.5pt; }
th, td { border: 1px solid #d0d0d0; padding: 5px 8px; text-align: left; }
th { background: #f4f4f6; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
em { color: #444; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 10pt; }
"""

def main():
    src = pathlib.Path(sys.argv[1]).resolve()
    out = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_suffix(".pdf")
    html_body = markdown.markdown(src.read_text(encoding="utf-8"),
                                  extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html); tmp = f.name
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", f"file://{tmp}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(tmp)
    print(f"PDF -> {out}")

if __name__ == "__main__":
    main()
