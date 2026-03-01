#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, sys

HEADER_RE = re.compile(r"\[window\d+-search-iter\d+-seg\d+\]")

def read_text(path):
    b = open(path, "rb").read()
    for enc in ("utf-8", "gbk"):
        try:
            s = b.decode(enc)
            if s.count("\ufffd") < 20:
                return s
        except Exception:
            pass
    return b.decode("utf-8", errors="ignore")

def main():
    if len(sys.argv) < 2:
        print("usage: python dump_search_iter_lines.py your.log")
        return
    path = sys.argv[1]
    txt = read_text(path)
    lines = txt.splitlines()
    hits = [ln for ln in lines if HEADER_RE.search(ln)]
    out = "search_iter_lines.txt"
    with open(out, "w", encoding="utf-8") as f:
        for ln in hits[:200]:
            f.write(ln + "\n")
    print(f"found {len(hits)} header lines, wrote first {min(200,len(hits))} to {out}")

if __name__ == "__main__":
    main()
