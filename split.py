#!/usr/bin/env python3
"""
Split the ICPC template .tex into individual code files and produce a new .tex
that uses \\lstinputlisting.  Python code blocks get .py extension.

Also patches \\lstset to add keepspaces=true for copy-friendly PDF output.

Produces:
  - templates/  directory with .cpp/.py files
  - order.txt   manifest of all extracted files
  - lib_split.tex  new tex file referencing the extracted code
"""

import os
import re
import shutil
import sys

TEX_SOURCE = 'SSerxhs 的 ICPC 模板.tex'
TEX_OUTPUT = 'lib_split.tex'
ORDER_FILE = 'order.txt'
TEMPLATE_DIR = 'templates'


def sanitize_filename(name):
    name = re.sub(r'\$[^$]*\$', lambda m: m.group(0).strip('$').replace('\\', '').replace('{', '').replace('}', ''), name)
    name = re.sub(r'\\verb\|([^|]*)\|', r'\1', name)
    name = re.sub(r'\\[a-zA-Z]+', '', name)
    name = re.sub(r'[{}]', '', name)
    name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    name = name.replace('（', '(').replace('）', ')')
    name = name.strip()
    return name


def is_python_section(section_stack):
    """Check if we are inside the 'python 使用方法' subsection."""
    for name in section_stack:
        if 'python' in name.lower():
            return True
    return False


def main():
    if not os.path.isfile(TEX_SOURCE):
        print(f'ERROR: {TEX_SOURCE} not found', file=sys.stderr)
        sys.exit(1)

    with open(TEX_SOURCE, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')

    # Clean old output
    if os.path.isdir(TEMPLATE_DIR):
        shutil.rmtree(TEMPLATE_DIR)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    section = ''
    subsection = ''
    subsubsection = ''

    section_re = re.compile(r'^\\section\{(.+?)\}')
    subsection_re = re.compile(r'^\\subsection\{(.+?)\}')
    subsubsection_re = re.compile(r'^\\subsubsection\{(.+?)\}')
    begin_lst = re.compile(r'^\\begin\{lstlisting\}')
    end_lst = re.compile(r'^\\end\{lstlisting\}')

    block_counter = {}
    new_lines = []
    in_lstlisting = False
    current_code = []
    order_list = []
    file_count = 0

    for line in lines:
        if not in_lstlisting:
            m = section_re.match(line)
            if m:
                section = sanitize_filename(m.group(1))
                subsection = ''
                subsubsection = ''
                new_lines.append(line)
                continue

            m = subsection_re.match(line)
            if m:
                subsection = sanitize_filename(m.group(1))
                subsubsection = ''
                new_lines.append(line)
                continue

            m = subsubsection_re.match(line)
            if m:
                subsubsection = sanitize_filename(m.group(1))
                new_lines.append(line)
                continue

        if not in_lstlisting and begin_lst.match(line):
            in_lstlisting = True
            current_code = []
            continue

        if in_lstlisting and end_lst.match(line):
            in_lstlisting = False

            # Determine directory and base name
            if subsubsection:
                dir_path = os.path.join(TEMPLATE_DIR, section, subsection)
                base_name = subsubsection
            elif subsection:
                dir_path = os.path.join(TEMPLATE_DIR, section)
                base_name = subsection
            elif section:
                dir_path = os.path.join(TEMPLATE_DIR, section)
                base_name = section
            else:
                dir_path = TEMPLATE_DIR
                base_name = 'misc'

            # Choose extension based on context
            py = is_python_section([section, subsection, subsubsection])
            ext = '.py' if py else '.cpp'

            # Handle multiple code blocks with same name
            key = (dir_path, base_name)
            if key not in block_counter:
                block_counter[key] = 0
            block_counter[key] += 1
            if block_counter[key] == 1:
                filename = base_name + ext
            else:
                filename = f'{base_name}_{block_counter[key]}{ext}'

            os.makedirs(dir_path, exist_ok=True)
            filepath = os.path.join(dir_path, filename)

            # Write code — exact content
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, code_line in enumerate(current_code):
                    f.write(code_line)
                    if i < len(current_code) - 1:
                        f.write('\n')
                if current_code:
                    f.write('\n')

            file_count += 1
            order_list.append(filepath)

            # In the tex, use lstinputlisting with language override for python
            if py:
                new_lines.append(f'\\lstinputlisting[language=python]{{{filepath}}}')
            else:
                new_lines.append(f'\\lstinputlisting{{{filepath}}}')

            current_code = []
            continue

        if in_lstlisting:
            current_code.append(line)
        else:
            new_lines.append(line)

    # --- Patch lstset: add keepspaces=true for copy-friendly PDF ---
    output = '\n'.join(new_lines)
    if 'keepspaces' not in output:
        output = output.replace(
            'columns=fullflexible,',
            'columns=fullflexible,\nkeepspaces=true,',
        )

    with open(TEX_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(output)

    # --- Write order.txt ---
    with open(ORDER_FILE, 'w', encoding='utf-8') as f:
        for p in order_list:
            f.write(p + '\n')

    print(f'Created {file_count} files in {TEMPLATE_DIR}/')
    print(f'Created {TEX_OUTPUT}')
    print(f'Created {ORDER_FILE} ({len(order_list)} entries)')

    # --- Verify ---
    with open(TEX_OUTPUT, 'r', encoding='utf-8') as f:
        out = f.read()
    n_input = out.count('\\lstinputlisting')
    n_begin = out.count('\\begin{lstlisting}')
    print(f'Verification: {n_input} \\lstinputlisting, {n_begin} remaining \\begin{{lstlisting}}')


if __name__ == '__main__':
    main()
