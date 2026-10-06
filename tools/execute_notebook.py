#!/usr/bin/env python3
"""Execute the notebook with the selected kernel and retain its checked outputs."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kernel", default="hnsc-methylation")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "notebooks" / "HNSC_Methylation_Pipeline.ipynb"
    book = nbformat.read(path, as_version=4)
    client = NotebookClient(book, timeout=1800, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(root / "notebooks")}},
                            allow_errors=False)
    client.execute()
    book.metadata["editor"] = "Musalula Sinkala"
    book.metadata["last_modified_by"] = "Musalula Sinkala"
    book.metadata["execution_checked"] = True
    nbformat.validate(book)
    nbformat.write(book, path)
    code = [c for c in book.cells if c.cell_type == "code"]
    print(f"Executed {len(code)} code cells without errors; saved {path.name}.")


if __name__ == "__main__":
    main()
