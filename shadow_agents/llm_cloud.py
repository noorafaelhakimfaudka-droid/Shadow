# shadow_agents/llm_cloud.py
"""
Klien Hybrid Cloud LLM Gratis: Google AI Studio (Gemini) + Groq Cloud.
- Google Gemini: Menggerakkan Auditor, Architect, Statistician, Feature Strategist (Analogi mendalam & narasi kaya).
- Groq Cloud: Menggerakkan Skeptic & Robustness Investigator (Kecepatan kilat 450+ token/s & kritik pedas 70B).
- Dilengkapi perisai failover silang dan fallback lokal Ollama.
"""

from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# =============================================================================
# 1. GOOGLE AI STUDIO (GEMINI FLASH)
# =============================================================================
@dataclass
class GeminiCloudConfig:
    api_key: str = ""
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.3
    max_output_tokens: int = 800
    timeout_seconds: int = 15

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)


class GeminiCloudClient:
    """Klien Google AI Studio untuk dialog edukatif, berbobot, dan kaya analogi."""

    def __init__(self, config: Optional[GeminiCloudConfig] = None):
        self.config = config or GeminiCloudConfig()
        self.candidate_models = [
            self.config.model,
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        system_prompt: str = "",
        format_json: bool = False,
        fallback_text: str = "",
        fallback_response: Optional[str] = None,
    ) -> str:
        fb = fallback_response if fallback_response is not None else fallback_text
        sys_p = system_instruction or system_prompt
        if not self.config.api_key:
            return fb

        contents = [{"parts": [{"text": prompt}]}]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }

        if sys_p:
            payload["systemInstruction"] = {
                "parts": [{"text": sys_p}]
            }

        if format_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        data = json.dumps(payload).encode("utf-8")

        for model_name in self.candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.config.api_key}"
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except Exception:
                continue

        return fb


# =============================================================================
# 2. GROQ CLOUD (LPUs ULTRA-FAST INFERENCE)
# =============================================================================
@dataclass
class GroqCloudConfig:
    api_key: str = ""
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.3
    max_output_tokens: int = 300
    timeout_seconds: int = 10

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)


class GroqCloudClient:
    """Klien Groq Cloud untuk inferensi instan sub-detik (Skeptic & Veto)."""

    def __init__(self, config: Optional[GroqCloudConfig] = None):
        self.config = config or GroqCloudConfig()
        self.candidate_models = [
            self.config.model,
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        system_prompt: str = "",
        format_json: bool = False,
        fallback_text: str = "",
        fallback_response: Optional[str] = None,
    ) -> str:
        fb = fallback_response if fallback_response is not None else fallback_text
        sys_p = system_instruction or system_prompt
        if not self.config.api_key:
            return fb

        messages = []
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        for model_name in self.candidate_models:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
            }
            if format_json:
                payload["response_format"] = {"type": "json_object"}

            data = json.dumps(payload).encode("utf-8")
            url = "https://api.groq.com/openai/v1/chat/completions"
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    choices = res_json.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
            except Exception:
                continue

        return fb


# =============================================================================
# 3. HYBRID MULTI-AGENT ROUTER (THE DREAM TEAM)
# =============================================================================
class HybridAgentRouter:
    """
    Router cerdas yang mengalokasikan beban kerja ke model yang paling cocok:
    - Skeptic & Robustness Investigator -> Groq (kecepatan kilat & kritik pedas).
    - Auditor, Architect, Statistician, Feature -> Gemini (analogi mendalam & konteks panjang).
    - Otomatis saling failover jika salah satu API limit atau offline.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiCloudClient] = None,
        groq_client: Optional[GroqCloudClient] = None,
    ):
        self.gemini = gemini_client or GeminiCloudClient()
        self.groq = groq_client or GroqCloudClient()

    def generate_agent_speech(
        self,
        agent_role: str,
        topic: str,
        context_data: str,
        fallback_quote: str = "",
    ) -> str:
        role_lower = agent_role.lower()

        # Strategi Pembagian Peran
        if "skeptic" in role_lower or "robustness" in role_lower or "veto" in role_lower:
            sys_prompt = (
                f"Kamu adalah {agent_role} di arena SHADOW. Sifatmu: sangat kritis, tidak mudah percaya angka mulus, "
                "mencurigai kebocoran data (target leakage), dan siap menggunakan hak VETO. "
                "Jawab dalam 1-2 kalimat tajam, pedas, analogis, dan langsung menusuk kelemahan model!"
            )
            # Coba Groq dulu (kilat), jika gagal beralih ke Gemini
            prompt = f"Topik: {topic}\nKonteks Data: {context_data}\nKeluarkan analisismu sekarang:"
            ans = self.groq.generate(prompt, system_instruction=sys_prompt, fallback_text="")
            if not ans:
                ans = self.gemini.generate(prompt, system_instruction=sys_prompt, fallback_text=fallback_quote)
            return ans or fallback_quote
        else:
            sys_prompt = (
                f"Kamu adalah {agent_role} di arena SHADOW ML. "
                "Tugasmu: berikan penjelasan data science yang panjang, detail, dan mudah dimengerti dengan analogi tajam membumi "
                "(misalnya analogi murid pemalas, brankas ujian sekolah, wasit pertandingan, dll). "
                "Hindari istilah kaku tanpa analogi. Buat narasi 2 kalimat yang memukau dan mendidik!"
            )
            prompt = f"Topik: {topic}\nKonteks Data: {context_data}\nKeluarkan narasi analisismu sekarang:"
            ans = self.gemini.generate(prompt, system_instruction=sys_prompt, fallback_text="")
            if not ans:
                ans = self.groq.generate(prompt, system_instruction=sys_prompt, fallback_text=fallback_quote)
            return ans or fallback_quote

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        system_instruction: str = "",
        format_json: bool = False,
        fallback_response: Optional[str] = None,
        fallback_text: str = "",
    ) -> str:
        """Metode serbaguna yang kompatibel dengan LocalLLMClient."""
        fb = fallback_response if fallback_response is not None else fallback_text
        sys_p = system_prompt or system_instruction
        p_lower = (prompt + " " + sys_p).lower()
        if any(k in p_lower for k in ["skeptic", "veto", "kanari", "robustness", "stres"]):
            ans = self.groq.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text="")
            if not ans:
                ans = self.gemini.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text=fb)
            return ans or fb
        else:
            ans = self.gemini.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text="")
            if not ans:
                ans = self.groq.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text=fb)
            return ans or fb
