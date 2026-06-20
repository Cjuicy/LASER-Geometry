import re
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_image_paths(data_path, sample_interval=1, extensions=IMAGE_EXTENSIONS):
    if sample_interval <= 0:
        raise ValueError("sample_interval must be a positive integer")

    data_dir = Path(data_path)
    image_paths = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    image_paths = sorted(image_paths, key=_natural_key)

    return [str(path) for path in image_paths[::sample_interval]]
