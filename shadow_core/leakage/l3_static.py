# shadow_core/leakage/l3_static.py
"""
L3 — Analisis Statis AST untuk Kebocoran Kode (PRD Bagian 7, 8.7, 12.1).
Menganalisis Abstract Syntax Tree (AST) dari kode Python/Notebook untuk mendeteksi
kebocoran pra-proses (fit_transform sebelum train_test_split, fit pada seluruh df).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import ast


@dataclass
class L3Finding:
    line_number: int
    pattern: str
    code_snippet: str
    is_blocker: bool
    description: str


@dataclass
class L3Report:
    passed: bool
    findings: List[L3Finding]
    limitations: str = (
        "Analisis statis AST bersifat best-effort pada alur sekuensial sederhana; "
        "tidak melacak aliasing dinamis objek, pemanggilan fungsi tak langsung, atau eksekusi berbasis dictionary."
    )


class LeakageASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.findings: List[L3Finding] = []
        self.seen_fit_transform = False
        self.seen_train_test_split = False
        self.fit_calls: List[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == "train_test_split":
            self.seen_train_test_split = True
            # Jika sebelumnya sudah ada fit/fit_transform pada objek scaler/imputer
            if self.seen_fit_transform:
                for line, snippet in self.fit_calls:
                    self.findings.append(
                        L3Finding(
                            line_number=line,
                            pattern="fit_before_split",
                            code_snippet=snippet,
                            is_blocker=True,
                            description=(
                                f"Deteksi kebocoran L3 di baris {line}: `{snippet}` dipanggil "
                                "sebelum `train_test_split`. Transformer mempelajari statistik seluruh dataset."
                            )
                        )
                    )
                self.fit_calls.clear()

        elif func_name in ("fit", "fit_transform"):
            # Cek apakah dipanggil pada variabel global / bukan train
            arg_names = []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    arg_names.append(arg.id)
            
            # Jika argumen adalah df, data, X (bukan X_train, train_df) dan belum split
            if not self.seen_train_test_split:
                self.seen_fit_transform = True
                snippet = f"{func_name}({', '.join(arg_names)})"
                self.fit_calls.append((node.lineno, snippet))

        self.generic_visit(node)


def analyze_code_for_leakage(code: str) -> L3Report:
    """Menganalisis kode Python secara statis menggunakan AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return L3Report(
            passed=False,
            findings=[
                L3Finding(
                    line_number=e.lineno or 0,
                    pattern="syntax_error",
                    code_snippet=str(e.text or ""),
                    is_blocker=True,
                    description=f"Kode tidak valid secara sintaksis: {e}"
                )
            ]
        )

    visitor = LeakageASTVisitor()
    visitor.visit(tree)

    passed = len(visitor.findings) == 0
    return L3Report(passed=passed, findings=visitor.findings)
