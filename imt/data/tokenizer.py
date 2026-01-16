"""Character-level tokenizer supporting full ASCII text."""

from typing import Dict, List

import mlx.core as mx


class ASCIITokenizer:
    """Character-level tokenizer for full ASCII text.

    Supports all printable ASCII characters (32-126) plus special tokens.
    This enables the model to work with any English text or code.

    Vocabulary:
    - Special tokens: PAD, UNK, BOS, EOS (0-3)
    - Printable ASCII characters (32-126): 95 characters
    - Tab and newline for code formatting

    Total vocab size: 100
    """

    def __init__(self) -> None:
        """Initialize tokenizer with character mappings."""
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Special tokens first
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for idx, token in enumerate(special_tokens):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

        # Add tab and newline explicitly (not in printable range but needed)
        offset = len(special_tokens)
        self.char_to_id["\t"] = offset
        self.id_to_char[offset] = "\t"
        offset += 1
        self.char_to_id["\n"] = offset
        self.id_to_char[offset] = "\n"
        offset += 1

        # Add all printable ASCII characters (32-126)
        # This includes: space, punctuation, digits, uppercase, lowercase
        for ascii_code in range(32, 127):
            char = chr(ascii_code)
            if char not in self.char_to_id:  # Skip if already added
                self.char_to_id[char] = offset
                self.id_to_char[offset] = char
                offset += 1

        self.vocab_size = len(self.char_to_id)
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input string to encode.

        Returns:
            List of token IDs.
        """
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        result = []
        for i in ids:
            char = self.id_to_char.get(i, "")
            # Skip special tokens in output except whitespace
            if char.startswith("<") and char.endswith(">"):
                continue
            result.append(char)
        return "".join(result)

    def encode_batch(self, texts: List[str], max_length: int) -> mx.array:
        """Encode and pad a batch of texts.

        Args:
            texts: List of strings to encode.
            max_length: Maximum sequence length (will pad/truncate to this).

        Returns:
            MLX array of shape (batch_size, max_length).
        """
        encoded = []
        for text in texts:
            ids = self.encode(text)[:max_length]
            # Pad to max_length
            ids = ids + [self.pad_id] * (max_length - len(ids))
            encoded.append(ids)
        return mx.array(encoded, dtype=mx.int32)

    def decode_batch(self, ids: mx.array) -> List[str]:
        """Decode a batch of token IDs to strings.

        Args:
            ids: MLX array of shape (batch_size, seq_len).

        Returns:
            List of decoded strings.
        """
        ids_list = ids.tolist()
        return [self.decode(seq) for seq in ids_list]

    def strip_padding(self, text: str) -> str:
        """Remove padding tokens from decoded text.

        Args:
            text: Decoded string potentially containing <PAD> tokens.

        Returns:
            String with padding removed.
        """
        return text.replace("<PAD>", "").strip()


# Alias for backwards compatibility
HashTokenizer = ASCIITokenizer
