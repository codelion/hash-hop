"""Character-level tokenizer for HashHop tasks."""

from typing import Dict, List

import mlx.core as mx


class HashTokenizer:
    """Simple character-level tokenizer for HashHop.

    Vocabulary:
    - Special tokens: PAD, UNK, BOS, EOS, EQUALS, QUOTE, SPACE, NEWLINE, GT
    - a-z (26)
    - A-Z (26)
    - 0-9 (10)

    Total vocab size: 71
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "=": 4,
        "'": 5,
        " ": 6,
        "\n": 7,
        ">": 8,  # For simplified KEY>VALUE format
    }

    def __init__(self) -> None:
        """Initialize tokenizer with character mappings."""
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Add special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

        # Add lowercase letters
        offset = len(self.SPECIAL_TOKENS)
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.char_to_id[c] = offset + i
            self.id_to_char[offset + i] = c

        # Add uppercase letters
        offset += 26
        for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            self.char_to_id[c] = offset + i
            self.id_to_char[offset + i] = c

        # Add digits
        offset += 26
        for i, c in enumerate("0123456789"):
            self.char_to_id[c] = offset + i
            self.id_to_char[offset + i] = c

        self.vocab_size = len(self.char_to_id)
        self.pad_id = self.SPECIAL_TOKENS["<PAD>"]
        self.unk_id = self.SPECIAL_TOKENS["<UNK>"]
        self.bos_id = self.SPECIAL_TOKENS["<BOS>"]
        self.eos_id = self.SPECIAL_TOKENS["<EOS>"]

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
        return "".join(self.id_to_char.get(i, "<UNK>") for i in ids)

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
        # Convert to Python list
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
