"""Embedded export interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Exporter(ABC):
    @abstractmethod
    def export(self, model, destination: str | Path) -> Path:
        raise NotImplementedError


class ONNXExporter(Exporter):
    def export(self, model, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("ONNX export placeholder for restricted neural model.\n", encoding="utf-8")
        return destination


class TFLiteExporter(Exporter):
    def export(self, model, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("TFLite export placeholder for restricted neural model.\n", encoding="utf-8")
        return destination


class GGUFExporter(Exporter):
    def export(self, model, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("GGUF experimental export placeholder for restricted neural model.\n", encoding="utf-8")
        return destination

