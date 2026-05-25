#!/usr/bin/env python3
"""
Build script: compile with xelatex, then clean up temp files.
"""

import os
import sys
import subprocess
import glob
import hashlib

TEX_FILE = "lib_split.tex"
MAX_PASSES = 5
# 定义需要清理的扩展名
CLEAN_EXTENSIONS = [
    "*.aux",
    "*.log",
    "*.out",
    "*.toc",
    "*.fls",
    "*.fdb_latexmk",
    "*.xdv",
    "*.gz",
]
WATCH_EXTENSIONS = [
    "*.aux",
    "*.toc",
    "*.out",
]
RERUN_HINTS = [
    "Rerun to get cross-references right",
    "Rerun to get outlines right",
    "Label(s) may have changed",
    "Table widths have changed",
]


def clean_temp_files():
    print("\n--- Cleaning up temporary files ---")
    for ext in CLEAN_EXTENSIONS:
        # glob.glob 可以匹配通配符，返回文件路径列表
        for file_path in glob.glob(ext):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}", file=sys.stderr)


def file_digest():
    h = hashlib.sha256()
    for ext in WATCH_EXTENSIONS:
        for file_path in sorted(glob.glob(ext)):
            h.update(file_path.encode())
            with open(file_path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def need_rerun():
    log_file = os.path.splitext(TEX_FILE)[0] + ".log"
    if not os.path.isfile(log_file):
        return True
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    return any(x in s for x in RERUN_HINTS)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- Compile ---
    if not os.path.isfile(TEX_FILE):
        print(f"ERROR: {TEX_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    last_digest = None
    for i in range(MAX_PASSES):
        print(f"\n--- xelatex pass {i+1} ---")
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", TEX_FILE], timeout=300
        )
        if r.returncode != 0:
            print(f"ERROR: xelatex exited with code {r.returncode}", file=sys.stderr)
            sys.exit(1)
        cur_digest = file_digest()
        if i > 0 and cur_digest == last_digest and not need_rerun():
            break
        last_digest = cur_digest
    else:
        print(f"ERROR: xelatex did not stabilize in {MAX_PASSES} passes", file=sys.stderr)
        sys.exit(1)

    print("\nBuild successful.")

    # --- Clean ---
    clean_temp_files()


if __name__ == "__main__":
    main()
