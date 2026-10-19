#!/usr/bin/env python3
"""Execute the self-contained analysis using this Python interpreter."""
from pathlib import Path
import os
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "code/notebooks/paper_results.ipynb"
    if not (root / "docs/paper-assets.csv").is_file():
        raise RuntimeError("The public package asset manifest is missing.")
    output = root / "results/paper_release_v1/exports/notebook_v2"
    output.mkdir(parents=True, exist_ok=True)
    runtime = root / ".runtime"
    for variable, subdir in [("JUPYTER_RUNTIME_DIR", "jupyter"),
                             ("IPYTHONDIR", "ipython"),
                             ("MPLCONFIGDIR", "matplotlib")]:
        path = runtime / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    notebook = nbformat.read(source, as_version=4)
    # Avoid silently executing in a different interpreter's registered kernel.
    client = NotebookClient(notebook, kernel_name="python3", timeout=600,
                            resources={"metadata": {"path": str(root)}},
                            allow_errors=False)
    manager = client.create_kernel_manager()
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    print(f"Python: {sys.executable}\nPackage: {root}", flush=True)
    try:
        client.execute(cwd=str(root))
    finally:
        # Preserve diagnostics without overwriting the editable source notebook.
        nbformat.write(notebook, output / "paper_results.executed.ipynb")
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (output / "paper_results.html").write_text(html, encoding="utf-8")
    print(f"Completed: {output / 'paper_results.html'}", flush=True)


if __name__ == "__main__":
    main()
