#!/usr/bin/env python3
"""
Build script: compile with xelatex, then clean up temp files.
"""

import os
import sys
import subprocess
import glob

TEX_FILE = "lib_split.tex"
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- Compile ---
    if not os.path.isfile(TEX_FILE):
        print(f"ERROR: {TEX_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    for i in range(2):
        print(f"\n--- xelatex pass {i+1} ---")
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", TEX_FILE], timeout=300
        )
        if r.returncode != 0:
            print(f"ERROR: xelatex exited with code {r.returncode}", file=sys.stderr)
            sys.exit(1)

    print("\nBuild successful.")

    # --- Clean ---
    clean_temp_files()


if __name__ == "__main__":
    main()
