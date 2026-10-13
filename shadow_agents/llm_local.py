# shadow_agents/llm_local.py
"""
Klien LLM Lokal (PRD 8.1, 12.2 T1.1, AC-15).
Aturan R12: Seluruhnya lokal, menolak host selain localhost/127.0.0.1.
Parameter tetap: temperature = 0, seed = 42, konteks terbatas 4096.
"""

from __future__ import annotations
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class LLMSecurityError(PermissionError):
    """Pelanggaran aturan R12 / koneksi jaringan ke luar localhost."""
    pass


class LLMServiceUnavailableError(RuntimeError):
    """Layanan LLM lokal mati atau tidak dapat diakses."""
    pass


@dataclass
class LLMConfig:
    host: str = "127.0.0.1:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.0
    seed: int = 42
    context_window: int = 4096
    max_output_tokens: int = 700
    timeout_seconds: int = 10


ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


class LocalLLMClient:
    def __init__(self, config: Optional[LLMConfig] = None, mock_response: Optional[str] = None):
        self.config = config or LLMConfig()
        self.mock_response = mock_response
        self._validate_host(self.config.host)
        self.resolved_model = self._detect_best_model()

    def _detect_best_model(self) -> str:
        """Mendeteksi model terbaik yang terpasang di Ollama (prioritas: qwen3.5:4b > qwen2.5:7b)."""
        if self.mock_response is not None:
            return self.config.model
        try:
            req = urllib.request.Request(f"http://{self.config.host}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                for preferred in ["qwen3.5:4b", "qwen3.5", "qwen2.5:3b", "qwen2.5:7b"]:
                    for m in models:
                        if preferred in m:
                            return m
                if models:
                    return models[0]
        except Exception:
            pass
        return self.config.model

    def _validate_host(self, host: str) -> None:
        clean_host = host.split("/")[0].strip()
        hostname = clean_host.split(":")[0].strip()
        if hostname not in ALLOWED_HOSTS:
            raise LLMSecurityError(
                f"Aturan R12 & AC-15 Dilanggar: Host '{host}' ditolak! "
                f"Hanya diizinkan ke localhost/127.0.0.1."
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        format_json: bool = True,
        fallback_response: Optional[str] = None,
    ) -> str:
        """Mengirim permintaan inferensi ke Ollama lokal."""
        self._validate_host(self.config.host)
        if self.mock_response is not None:
            return self.mock_response

        url = f"http://{self.config.host}/api/generate"
        payload = {
            "model": getattr(self, "resolved_model", self.config.model),
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_ctx": self.config.context_window,
                "num_predict": self.config.max_output_tokens,
            },
        }
        if format_json:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                return resp_json.get("response", "")
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError) as e:
            if fallback_response is not None:
                return fallback_response
            raise LLMServiceUnavailableError(
                f"Layanan LLM lokal pada '{self.config.host}' tidak dapat diakses: {e}"
            ) from e
