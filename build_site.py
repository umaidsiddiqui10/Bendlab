#!/usr/bin/env python3
"""Build the GitHub Pages site in docs/ from the standalone pages.

Adds a shared tab bar (Simulator / Trained model / Curve fitter) to:
  bendlab_visual_simulator.html -> docs/index.html
  bendlab_model_demo.html       -> docs/model.html
  bendlab_curve_fitter.html     -> docs/fitter.html
Run after editing any of the standalone pages.
"""
from __future__ import annotations

NAV_CSS = """
  .site-nav{max-width:1180px;margin:0 auto;padding:14px 28px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .site-nav .brand{font-weight:800;font-size:15px;margin-right:8px}
  .site-nav a{text-decoration:none;color:#41544a;font-size:13.5px;font-weight:600;
    padding:7px 14px;border:1px solid #dce5dd;border-radius:999px;background:#ffffff}
  .site-nav a.active{background:#177245;border-color:#177245;color:#ffffff}
  .site-nav a:hover{border-color:#177245}
  .site-nav .spacer{flex:1}
"""

PAGES = [
    ("bendlab_visual_simulator.html", "docs/index.html", "simulator", "1480px"),
    ("bendlab_model_demo.html", "docs/model.html", "model", "1180px"),
    ("bendlab_curve_fitter.html", "docs/fitter.html", "fitter", "1180px"),
]


def nav(active: str, width: str) -> str:
    def cls(key: str) -> str:
        return ' class="active"' if key == active else ""
    return (
        f'<nav class="site-nav" style="max-width:{width}">'
        '<span class="brand">BendLab</span>'
        f'<a{cls("simulator")} href="index.html">Simulator</a>'
        f'<a{cls("model")} href="model.html">Trained model</a>'
        f'<a{cls("fitter")} href="fitter.html">Curve fitter</a>'
        '<span class="spacer"></span>'
        '<a href="https://github.com/Yash-Tripath1/Bendlab">GitHub</a>'
        "</nav>"
    )


def main() -> None:
    for src, dst, key, width in PAGES:
        html = open(src, encoding="utf-8").read()
        assert "</style>" in html and "<body>" in html, src
        html = html.replace("</style>", NAV_CSS + "</style>", 1)
        html = html.replace("<body>", "<body>\n" + nav(key, width), 1)
        open(dst, "w", encoding="utf-8").write(html)
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
