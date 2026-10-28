"""Свободная видеопамять NVIDIA — чтобы не грузить Ollama поверх игры."""
import logging
import shutil
import subprocess

log = logging.getLogger("capri")
NO_WINDOW = 0x08000000

# сколько видеопамяти обычно занимает модель Ollama в квантовании Q4
VRAM_BY_SIZE = (
    ("14b", 9200),
    ("8b", 5500),
    ("7b", 5000),
    ("4b", 2800),
    ("3b", 2200),
    ("1.7b", 1600),
    ("1.5b", 1500),
)


def model_vram_mb(name):
    """Оценка видеопамяти модели в мегабайтах (веса + короткий контекст)."""
    n = (name or "").lower()
    for tag, mb in VRAM_BY_SIZE:
        if tag in n:
            return mb
    return 4000


def vram():
    """(занято МБ, всего МБ, свободно МБ) или None, если NVIDIA / nvidia-smi нет."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, creationflags=NO_WINDOW,
        ).stdout.strip()
        if not out:
            return None
        best = None
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            used, total, free = int(parts[0]), int(parts[1]), int(parts[2])
            if best is None or total > best[1]:
                best = (used, total, free)
        return best
    except Exception:
        log.info("Не удалось узнать видеопамять через nvidia-smi")
        return None
