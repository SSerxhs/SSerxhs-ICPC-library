#!/usr/bin/env python3
"""
Build script: validate order.txt vs disk, then compile with xelatex.
"""

import os
import sys
import subprocess

TEX_FILE = 'lib_split.tex'
ORDER_FILE = 'order.txt'
TEMPLATE_DIR = 'templates'


def collect_files(root):
    result = set()
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.cpp') or f.endswith('.py'):
                result.add(os.path.join(dirpath, f))
    return result


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- Validate ---
    if not os.path.isfile(ORDER_FILE):
        print(f'ERROR: {ORDER_FILE} not found. Run split.py first.', file=sys.stderr)
        sys.exit(1)

    with open(ORDER_FILE, 'r', encoding='utf-8') as f:
        order_list = [line.strip() for line in f if line.strip()]

    order_set = set(order_list)
    if len(order_set) != len(order_list):
        seen = set()
        for p in order_list:
            if p in seen:
                print(f'ERROR: duplicate in {ORDER_FILE}: {p}', file=sys.stderr)
            seen.add(p)
        sys.exit(1)

    disk_set = collect_files(TEMPLATE_DIR)

    missing = order_set - disk_set
    orphan = disk_set - order_set
    ok = True

    if missing:
        ok = False
        print(f'ERROR: in {ORDER_FILE} but NOT on disk:', file=sys.stderr)
        for p in sorted(missing):
            print(f'  - {p}', file=sys.stderr)

    if orphan:
        ok = False
        print(f'ERROR: on disk but NOT in {ORDER_FILE}:', file=sys.stderr)
        for p in sorted(orphan):
            print(f'  + {p}', file=sys.stderr)

    if not ok:
        sys.exit(1)

    print(f'Validation passed: {len(order_list)} files consistent.')

    # --- Compile ---
    if not os.path.isfile(TEX_FILE):
        print(f'ERROR: {TEX_FILE} not found. Run split.py first.', file=sys.stderr)
        sys.exit(1)

    for i in range(2):
        print(f'\n--- xelatex pass {i+1} ---')
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', TEX_FILE], timeout=300)
        if r.returncode != 0:
            print(f'ERROR: xelatex exited with code {r.returncode}', file=sys.stderr)
            sys.exit(1)

    print('\nBuild successful.')


if __name__ == '__main__':
    main()
