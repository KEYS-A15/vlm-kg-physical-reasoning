from __future__ import annotations

__all__ = ["__version__", "main"]
__version__ = "0.1.0"


def main() -> None:
    from vlm_kg_physical_reasoning.cli import main as cli_main

    cli_main()
