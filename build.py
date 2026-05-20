#!/usr/bin/env python3
"""
Build script: compile with xelatex.
"""

import os
import sys
import subprocess

TEX_FILE = "lib_split.tex"


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


if __name__ == "__main__":
    main()
