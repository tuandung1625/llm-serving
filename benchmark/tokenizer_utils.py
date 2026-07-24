from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from benchmark.schemas import ChatMessage

LOGGER = logging.getLogger(__name__)


class TokenCounter(Protocol):
    name: str

    def count_text(self, text: str) -> int:
        ...

    def count_messages(self, messages: list[ChatMessage]) -> int:
        ...

    def synthetic_text(self, label: str, target_tokens: int) -> str:
        ...


class WhitespaceTokenCounter:
    name = "whitespace-fallback"

    def count_text(self, text: str) -> int:
        return len(text.split()) if text else 0

    def count_messages(self, messages: list[ChatMessage]) -> int:
        return sum(self.count_text(f"{message.get('role', '')}: {message.get('content', '')}") for message in messages)

    def synthetic_text(self, label: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        return " ".join(f"{label}_{index:05d}" for index in range(target_tokens))


class HuggingFaceTokenCounter:
    def __init__(self, model_path: str | Path):
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model path does not exist: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.name = f"huggingface:{self.model_path}"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: list[ChatMessage]) -> int:
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("chat_template_token_count_failed", extra={"error": str(exc)})
            rendered = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            return self.count_text(rendered)

    def synthetic_text(self, label: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        seed_words = [f"{label}_{index % 997:03d}" for index in range(max(target_tokens * 3, 8))]
        text = " ".join(seed_words)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        while len(token_ids) < target_tokens:
            text = f"{text} {text}"
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        candidate = self.tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=True)
        if self.count_text(candidate) == 0:
            return " ".join(seed_words[:target_tokens])
        return candidate


def load_token_counter(model_path: str | Path, *, required: bool = True) -> TokenCounter:
    try:
        return HuggingFaceTokenCounter(model_path)
    except Exception:
        if required:
            raise
        LOGGER.warning("using_whitespace_tokenizer_fallback", extra={"model_path": str(model_path)})
        return WhitespaceTokenCounter()

