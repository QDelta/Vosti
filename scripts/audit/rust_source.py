"""Mask Rust comments and literals while preserving source offsets and lines.

This is lexical support for the audit scanner, not a Rust parser. In particular,
lifetimes remain code; quoted character literals and nested block comments do
not contribute braces to declaration or module nesting.
"""

import re


_RAW = re.compile(r'(?:br|r)(?P<hashes>\#*)"')
_CHAR = re.compile(r"'(?:\\u\{[0-9a-fA-F_]+\}|\\x[0-9a-fA-F]{2}|\\[^\n]|[^'\\\n])'")


def code_only(source: str) -> str:
    result = list(source)
    cursor = 0
    while cursor < len(source):
        start = cursor
        if source.startswith("//", cursor):
            end = source.find("\n", cursor)
            cursor = len(source) if end < 0 else end
        elif source.startswith("/*", cursor):
            cursor += 2
            depth = 1
            while cursor < len(source) and depth:
                if source.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif source.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            if depth:
                raise ValueError("unterminated Rust block comment")
        else:
            raw = _RAW.match(source, cursor) if cursor == 0 or not (
                source[cursor - 1].isalnum() or source[cursor - 1] == "_") else None
            char = _CHAR.match(source, cursor)
            if raw:
                delimiter = '"' + raw["hashes"]
                end = source.find(delimiter, raw.end())
                if end < 0:
                    raise ValueError("unterminated Rust raw string")
                cursor = end + len(delimiter)
            elif char:
                cursor = char.end()
            elif source[cursor] == '"':
                cursor += 1
                while cursor < len(source) and source[cursor] != '"':
                    cursor += 2 if source[cursor] == "\\" else 1
                if cursor >= len(source):
                    raise ValueError("unterminated Rust string")
                cursor += 1
            else:
                cursor += 1
                continue
        for index in range(start, cursor):
            if source[index] not in "\r\n":
                result[index] = " "
    return "".join(result)
