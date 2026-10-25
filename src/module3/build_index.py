"""Build or rebuild the saved Module 3 image index in one run."""
from __future__ import annotations

from .retrieval import build_index


def main() -> None:
    stats = build_index()
    print("Module 3 image index built successfully.")
    for name, value in stats.items():
        print(f"{name.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()
