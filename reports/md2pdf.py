"""
md2pdf.py -- minimal Markdown -> LaTeX -> PDF converter for the analysis reports.

Handles the subset of Markdown the report generators emit: ``#``/``##``/``###``
headers, ``**bold**``, `` `code` ``, ``- `` bullet lists, ``| ... |`` tables,
``![alt](path)`` images, and paragraphs.  Unicode (Greek, arrows, super-scripts,
etc.) is mapped to LaTeX commands so it compiles with plain ``pdflatex``.

Usage:
    from md2pdf import md_to_pdf
    md_to_pdf(markdown_text, "/abs/path/out.pdf", base_dir="/abs/dir")
"""

import os
import re
import shutil
import subprocess

_GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "Γ": r"\Gamma", "δ": r"\delta",
    "Δ": r"\Delta", "ε": r"\epsilon", "η": r"\eta", "θ": r"\theta", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "µ": r"\mu", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi", "χ": r"\chi", "ω": r"\omega",
}
_SYM = {
    "⁰": r"^{0}", "¹": r"^{1}", "²": r"^{2}", "³": r"^{3}", "⁴": r"^{4}",
    "⁵": r"^{5}", "⁶": r"^{6}", "⁷": r"^{7}", "⁸": r"^{8}", "⁹": r"^{9}",
    "⁻": r"^{-}",
    "₀": r"_{0}", "₁": r"_{1}", "₂": r"_{2}", "₃": r"_{3}", "₄": r"_{4}",
    "₅": r"_{5}", "₆": r"_{6}", "₇": r"_{7}", "₈": r"_{8}", "₉": r"_{9}",
    "×": r"\times", "·": r"\cdot",
    "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "≈": r"\approx", "≪": r"\ll", "≫": r"\gg", "≤": r"\le", "≥": r"\ge",
    "≲": r"\lesssim", "≳": r"\gtrsim", "≠": r"\neq",
    "∝": r"\propto", "±": r"\pm", "∼": r"\sim", "√": r"\surd", "≅": r"\cong",
    "⟨": r"\langle ", "⟩": r"\rangle",
}


# Science tokens written in ASCII: a single-letter base (not mid-word; a leading
# digit coefficient like 6D_r is fine) followed by _/^ sub/superscripts.  These
# are converted to proper LaTeX math; multi-letter scripts (app, min, VV) are set
# roman.  Identifiers/filenames (size_max, swy3_report, litfsi_raw) have word
# bases and are left untouched.
_SCRIPT = re.compile(r'([_^])(\{[^}]*\}|[−+\-]?[A-Za-z0-9](?:[A-Za-z0-9.,]*[A-Za-z0-9])?)')
_MATH_TOKEN = re.compile(
    r'(?<![A-Za-z\\])([A-Za-z])'
    r'((?:[_^](?:\{[^}]*\}|[−+\-]?[A-Za-z0-9](?:[A-Za-z0-9.,]*[A-Za-z0-9])?))+)')


def _scripts(part: str) -> str:
    out = []
    for op, val in _SCRIPT.findall(part):
        inner = val[1:-1] if val.startswith("{") else val
        inner = inner.replace("−", "-")
        if sum(c.isalpha() for c in inner) >= 2:   # words (app, min, VV) -> roman
            inner = r"\mathrm{" + inner + "}"
        out.append(f"{op}{{{inner}}}")
    return "".join(out)


def _mathify(s: str, keep) -> str:
    s = re.sub(r"(?<![A-Za-z])k_BT\b", lambda m: keep(r"$k_\mathrm{B}T$"), s)
    s = re.sub(r"(?<![A-Za-z])k_B\b", lambda m: keep(r"$k_\mathrm{B}$"), s)
    return _MATH_TOKEN.sub(lambda m: keep("$" + m.group(1) + _scripts(m.group(2)) + "$"), s)


def _uni(s: str) -> str:
    for u, t in _GREEK.items():
        s = s.replace(u, f"${t}$")
    for u, t in _SYM.items():
        s = s.replace(u, f"${t}$")
    s = s.replace("−", "-").replace("–", "--").replace("—", "---")
    s = s.replace("°", r"$^\circ$").replace("…", r"\ldots")
    # plain '<' and '>' render as inverted punctuation in LaTeX text mode
    s = s.replace("->", r"$\to$").replace(">", r"$>$").replace("<", r"$<$")
    return s


def _esc(s: str) -> str:
    out = []
    for ch in s:
        if ch in "%&#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _textseg(s: str) -> str:
    # 1. pull out science tokens as ready-made LaTeX math (protected from escaping)
    stash = []

    def keep(latex):
        stash.append(latex); return f"\x00{len(stash) - 1}\x00"

    s = _mathify(s, keep)
    # 2. escape + map the remaining plain text, then 3. restore the math
    s = _uni(_esc(s))
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)


def _inline(s: str) -> str:
    # code spans, then bold, then italic, then plain
    parts = re.split(r"(`[^`]*`|\*\*[^*]+\*\*|\*[^*]+\*)", s)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`"):
            out.append(r"\texttt{" + _textseg(p[1:-1]) + "}")
        elif p.startswith("**") and p.endswith("**"):
            out.append(r"\textbf{" + _textseg(p[2:-2]) + "}")
        elif p.startswith("*") and p.endswith("*") and len(p) > 2:
            out.append(r"\emph{" + _textseg(p[1:-1]) + "}")
        else:
            out.append(_textseg(p))
    return "".join(out)


def _table(rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = rows[2:]
    ncol = len(header)
    # first column left-aligned, remaining columns wrap (tabularx X)
    colspec = "l" + "X" * (ncol - 1) if ncol > 1 else "X"
    lines = [r"\begin{center}\small",
             r"\begin{tabularx}{\textwidth}{" + colspec + "}",
             r"\toprule",
             " & ".join(_inline(h) for h in header) + r" \\",
             r"\midrule"]
    for r in body:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        cells = (cells + [""] * ncol)[:ncol]
        lines.append(" & ".join(_inline(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return lines


def md_to_latex(md: str, title: str = None) -> str:
    lines = md.split("\n")
    body, i, in_list = [], 0, False

    def close_list():
        nonlocal in_list
        if in_list:
            body.append(r"\end{itemize}"); in_list = False

    while i < len(lines):
        ln = lines[i]
        # tables
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_list()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i]); i += 1
            body += _table(block); continue
        m_img = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln.strip())
        if m_img:
            close_list()
            body.append(r"\begin{center}\includegraphics[width=0.98\textwidth]{"
                        + m_img.group(1) + r"}\end{center}")
            i += 1; continue
        if ln.startswith("### "):
            close_list(); body.append(r"\subsubsection*{" + _inline(ln[4:]) + "}")
        elif ln.startswith("## "):
            close_list(); body.append(r"\subsection*{" + _inline(ln[3:]) + "}")
        elif ln.startswith("# "):
            close_list(); body.append(r"\section*{" + _inline(ln[2:]) + "}")
        elif ln.strip().startswith("- "):
            if not in_list:
                body.append(r"\begin{itemize}\setlength\itemsep{1pt}"); in_list = True
            body.append(r"\item " + _inline(ln.strip()[2:]))
        elif ln.strip() == "":
            close_list(); body.append("")
        else:
            close_list(); body.append(_inline(ln) + r"\\")
        i += 1
    close_list()

    preamble = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[letterpaper,margin=1.8cm]{geometry}",
        r"\usepackage{graphicx,booktabs,amsmath,amssymb,tabularx}",
        r"\usepackage[colorlinks=true,allcolors=blue]{hyperref}",
        r"\usepackage{parskip}",
        r"\setcounter{secnumdepth}{0}",
        r"\begin{document}",
    ]
    return "\n".join(preamble + body + [r"\end{document}"])


def md_to_pdf(md: str, out_pdf: str, base_dir: str = None) -> str:
    base_dir = base_dir or os.path.dirname(os.path.abspath(out_pdf))
    tex = md_to_latex(md)
    stem = os.path.splitext(os.path.basename(out_pdf))[0]
    tex_path = os.path.join(base_dir, stem + ".tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    log = ""
    for _ in range(2):
        p = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             os.path.basename(tex_path)],
            cwd=base_dir, capture_output=True, text=True)
        log = p.stdout + p.stderr
    pdf_path = os.path.join(base_dir, stem + ".pdf")
    # clean aux
    for ext in (".aux", ".log", ".out", ".tex"):
        try:
            os.remove(os.path.join(base_dir, stem + ext))
        except OSError:
            pass
    if not os.path.exists(pdf_path):
        raise RuntimeError("pdflatex failed:\n" + log[-2000:])
    if out_pdf != pdf_path:
        shutil.move(pdf_path, out_pdf)
    return out_pdf
