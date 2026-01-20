"""Hybrid Code Tokenizer.

Combines:
1. Symbol tokenization (functions, variables, classes as single tokens)
2. Syntax tokenization (keywords, operators, literals)

This enables:
- Perfect symbol retrieval (like our HashHop solution)
- Compact representation (fewer tokens for same code)
- Better long-context handling
"""

import ast
import re
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Token:
    """A single token with type and value."""
    type: str      # SYMBOL, KEYWORD, OP, LITERAL, INDENT, NEWLINE, etc.
    value: str     # The actual string value
    id: int = -1   # Token ID after encoding


@dataclass
class CodeTokenizer:
    """Hybrid tokenizer for Python code.

    Symbols (function names, variables, classes) become single tokens.
    Syntax elements use a fixed vocabulary.
    """

    # Fixed vocabulary for syntax
    KEYWORDS: Set[str] = field(default_factory=lambda: {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
        'while', 'with', 'yield'
    })

    OPERATORS: Set[str] = field(default_factory=lambda: {
        '+', '-', '*', '/', '//', '%', '**', '@',
        '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', '@=',
        '==', '!=', '<', '>', '<=', '>=',
        '&', '|', '^', '~', '<<', '>>', '&=', '|=', '^=', '<<=', '>>=',
        '(', ')', '[', ']', '{', '}',
        ',', ':', ';', '.', '->', '...', '\\',
        "'", '"', '#'
    })

    SPECIAL_TOKENS: Dict[str, int] = field(default_factory=lambda: {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
        '<INDENT>': 4,
        '<DEDENT>': 5,
        '<NEWLINE>': 6,
        '<CONTEXT>': 7,
        '</CONTEXT>': 8,
        '<INSTRUCTION>': 9,
        '</INSTRUCTION>': 10,
        '<RESPONSE>': 11,
        '</RESPONSE>': 12,
    })

    def __post_init__(self):
        # Build vocabulary
        self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        next_id = len(self.SPECIAL_TOKENS)

        # Add keywords
        for kw in sorted(self.KEYWORDS):
            self.token_to_id[f'KW:{kw}'] = next_id
            self.id_to_token[next_id] = f'KW:{kw}'
            next_id += 1

        # Add operators
        for op in sorted(self.OPERATORS):
            self.token_to_id[f'OP:{op}'] = next_id
            self.id_to_token[next_id] = f'OP:{op}'
            next_id += 1

        # Symbol IDs start after fixed vocabulary
        self.symbol_start_id = next_id
        self.symbols: Dict[str, int] = {}  # symbol_name -> id
        self.next_symbol_id = next_id

    def _add_symbol(self, name: str) -> int:
        """Add a symbol to vocabulary, return its ID."""
        if name not in self.symbols:
            self.symbols[name] = self.next_symbol_id
            self.token_to_id[f'SYM:{name}'] = self.next_symbol_id
            self.id_to_token[self.next_symbol_id] = f'SYM:{name}'
            self.next_symbol_id += 1
        return self.symbols[name]

    def _extract_symbols(self, code: str) -> Set[str]:
        """Extract all symbols (names) from Python code using AST."""
        symbols = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    symbols.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    symbols.add(node.name)
                    for arg in node.args.args:
                        symbols.add(arg.arg)
                elif isinstance(node, ast.AsyncFunctionDef):
                    symbols.add(node.name)
                    for arg in node.args.args:
                        symbols.add(arg.arg)
                elif isinstance(node, ast.ClassDef):
                    symbols.add(node.name)
                elif isinstance(node, ast.arg):
                    symbols.add(node.arg)
                elif isinstance(node, ast.alias):
                    if node.asname:
                        symbols.add(node.asname)
                    else:
                        # For 'import foo.bar', add 'foo'
                        symbols.add(node.name.split('.')[0])
                elif isinstance(node, ast.Attribute):
                    # Capture attribute names like obj.price
                    symbols.add(node.attr)
        except SyntaxError:
            # Fall back to regex for invalid Python
            symbols = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))

        # Filter out keywords
        symbols = symbols - self.KEYWORDS
        return symbols

    def tokenize(self, code: str) -> List[Token]:
        """Tokenize Python code into hybrid tokens."""
        # First, extract all symbols
        symbols = self._extract_symbols(code)
        for sym in symbols:
            self._add_symbol(sym)

        tokens = []
        lines = code.split('\n')

        for line_no, line in enumerate(lines):
            # Handle indentation
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 0:
                # Represent indentation as number of spaces/4
                indent_level = indent // 4
                for _ in range(indent_level):
                    tokens.append(Token('INDENT', '    ', self.token_to_id['<INDENT>']))

            if not stripped:
                tokens.append(Token('NEWLINE', '\n', self.token_to_id['<NEWLINE>']))
                continue

            # Handle comments
            if stripped.startswith('#'):
                # Keep comments as single tokens
                tokens.append(Token('COMMENT', stripped, self._add_symbol(stripped)))
                tokens.append(Token('NEWLINE', '\n', self.token_to_id['<NEWLINE>']))
                continue

            # Tokenize the line
            pos = 0
            text = stripped

            while pos < len(text):
                # Skip whitespace
                if text[pos].isspace():
                    pos += 1
                    continue

                # Check for string literals
                if text[pos] in '"\'':
                    quote = text[pos]
                    # Check for triple quotes
                    if text[pos:pos+3] in ('"""', "'''"):
                        end = text.find(text[pos:pos+3], pos+3)
                        if end == -1:
                            literal = text[pos:]
                            pos = len(text)
                        else:
                            literal = text[pos:end+3]
                            pos = end + 3
                    else:
                        # Single quote string
                        end = pos + 1
                        while end < len(text):
                            if text[end] == '\\':
                                end += 2
                            elif text[end] == quote:
                                end += 1
                                break
                            else:
                                end += 1
                        literal = text[pos:end]
                        pos = end
                    tokens.append(Token('STRING', literal, self._add_symbol(literal)))
                    continue

                # Check for numbers
                if text[pos].isdigit() or (text[pos] == '.' and pos+1 < len(text) and text[pos+1].isdigit()):
                    end = pos
                    while end < len(text) and (text[end].isdigit() or text[end] in '.eExXoObBjJ_'):
                        end += 1
                    number = text[pos:end]
                    tokens.append(Token('NUMBER', number, self._add_symbol(number)))
                    pos = end
                    continue

                # Check for identifiers (symbols or keywords)
                if text[pos].isalpha() or text[pos] == '_':
                    end = pos
                    while end < len(text) and (text[end].isalnum() or text[end] == '_'):
                        end += 1
                    word = text[pos:end]

                    # Check for f-string prefix
                    if word in ('f', 'r', 'b', 'fr', 'rf', 'br', 'rb') and end < len(text) and text[end] in '"\'':
                        # This is a string prefix, handle as part of string literal
                        quote = text[end]
                        # Check for triple quotes
                        if text[end:end+3] in ('"""', "'''"):
                            str_end = text.find(text[end:end+3], end+3)
                            if str_end == -1:
                                literal = text[pos:]
                                pos = len(text)
                            else:
                                literal = text[pos:str_end+3]
                                pos = str_end + 3
                        else:
                            # Single quote string
                            str_end = end + 1
                            while str_end < len(text):
                                if text[str_end] == '\\':
                                    str_end += 2
                                elif text[str_end] == quote:
                                    str_end += 1
                                    break
                                else:
                                    str_end += 1
                            literal = text[pos:str_end]
                            pos = str_end
                        tokens.append(Token('STRING', literal, self._add_symbol(literal)))
                        continue

                    if word in self.KEYWORDS:
                        tokens.append(Token('KEYWORD', word, self.token_to_id[f'KW:{word}']))
                    elif word in self.symbols:
                        tokens.append(Token('SYMBOL', word, self.symbols[word]))
                    else:
                        # Symbol not in AST-extracted set, add it dynamically
                        self._add_symbol(word)
                        tokens.append(Token('SYMBOL', word, self.symbols[word]))
                    pos = end
                    continue

                # Check for operators (multi-char first)
                matched = False
                for length in [3, 2, 1]:
                    op = text[pos:pos+length]
                    if op in self.OPERATORS:
                        tokens.append(Token('OP', op, self.token_to_id[f'OP:{op}']))
                        pos += length
                        matched = True
                        break

                if not matched:
                    # Unknown character, treat as symbol
                    tokens.append(Token('UNK', text[pos], self.token_to_id['<UNK>']))
                    pos += 1

            tokens.append(Token('NEWLINE', '\n', self.token_to_id['<NEWLINE>']))

        return tokens

    def encode(self, code: str) -> List[int]:
        """Encode code to token IDs."""
        tokens = self.tokenize(code)
        return [t.id for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to code."""
        result = []
        for id in ids:
            token_str = self.id_to_token.get(id, '<UNK>')

            if token_str == '<INDENT>':
                result.append('    ')
            elif token_str == '<NEWLINE>':
                result.append('\n')
            elif token_str.startswith('KW:'):
                result.append(token_str[3:] + ' ')
            elif token_str.startswith('OP:'):
                result.append(token_str[3:])
            elif token_str.startswith('SYM:'):
                result.append(token_str[4:])
            elif token_str in ('<PAD>', '<UNK>', '<BOS>', '<EOS>'):
                pass
            else:
                result.append(token_str)

        return ''.join(result)

    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return self.next_symbol_id

    def save(self, path: str):
        """Save tokenizer state."""
        state = {
            'symbols': self.symbols,
            'next_symbol_id': self.next_symbol_id,
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        """Load tokenizer state."""
        with open(path) as f:
            state = json.load(f)

        self.symbols = state['symbols']
        self.next_symbol_id = state['next_symbol_id']

        # Rebuild token_to_id and id_to_token for symbols
        for name, id in self.symbols.items():
            self.token_to_id[f'SYM:{name}'] = id
            self.id_to_token[id] = f'SYM:{name}'


def demo():
    """Demonstrate the tokenizer."""
    code = '''
def calculate_total(items):
    """Calculate the total price of items."""
    total = 0
    for item in items:
        total += item.price
    return total

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def get_total(self):
        return calculate_total(self.items)
'''

    tokenizer = CodeTokenizer()
    tokens = tokenizer.tokenize(code)

    print("=" * 60)
    print("HYBRID CODE TOKENIZER DEMO")
    print("=" * 60)
    print(f"\nOriginal code length: {len(code)} chars")
    print(f"Token count: {len(tokens)}")
    print(f"Vocabulary size: {tokenizer.vocab_size()}")

    # Compare with simple whitespace tokenization
    simple_tokens = code.split()
    print(f"Simple split tokens: {len(simple_tokens)}")
    print(f"Compression ratio: {len(simple_tokens) / len(tokens):.2f}x")

    print("\n" + "-" * 60)
    print("Sample tokens:")
    print("-" * 60)
    for t in tokens[:30]:
        print(f"  {t.type:10} | {t.id:5} | {repr(t.value)}")

    print("\n" + "-" * 60)
    print("Symbols extracted:")
    print("-" * 60)
    for name, id in sorted(tokenizer.symbols.items(), key=lambda x: x[1]):
        if not name.startswith('#') and not name.startswith('"'):
            print(f"  {id:5} | {name}")

    # Test encode/decode roundtrip
    print("\n" + "-" * 60)
    print("Encode/Decode roundtrip:")
    print("-" * 60)
    ids = tokenizer.encode(code)
    decoded = tokenizer.decode(ids)
    print(f"Original == Decoded: {code.strip() == decoded.strip()}")


if __name__ == '__main__':
    demo()
