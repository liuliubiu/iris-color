"""Convert brand PNG / misnamed ICO to valid Windows icon.ico for electron-builder."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "brand"
OUT = ROOT / "iris-desktop" / "build" / "icon.ico"

SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]


def pick_source() -> Path | None:
    for name in ("app-icon.ico", "logo.png", "favicon.png"):
        path = BRAND / name
        if path.exists():
            return path
    return None


def is_png(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(8) == b"\x89PNG\r\n\x1a\n"


def to_square_icon(img: Image.Image, size: int = 256) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2), resized)
    return canvas


def convert(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    square = to_square_icon(Image.open(src), 256)
    square.save(dest, format="ICO", sizes=SIZES)
    print(f"converted: {src} -> {dest} ({square.size[0]}x{square.size[1]})")


def main() -> None:
    src = pick_source()
    if not src:
        print("skip: no app-icon.ico / logo.png in brand/")
        return
    if src.suffix.lower() == ".ico" and not is_png(src):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(src.read_bytes())
        print(f"copied valid ico: {src} -> {OUT}")
        return
    convert(src, OUT)


if __name__ == "__main__":
    main()
