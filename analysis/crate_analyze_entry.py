"""crate_analyze_entry.py — PyInstaller entry for the bundled analysis executable (`crate-analyze`).

In the self-contained build the heavy pipeline (torch + MuQ + librosa + umap …) is frozen into its
OWN exe so the GUI process stays light and a crashing analysis step can't take the window down. This
entry is a tiny dispatcher:

    crate-analyze [--root … --limit … --rebuild …]   -> the full pipeline (analyze_all.main)
    crate-analyze --step <module> [args]             -> just that one step's main()

`analyze_all`, when frozen, drives the pipeline by re-invoking THIS exe once per step with `--step`
(so each heavy step runs in its own process and frees its memory afterward — same isolation the
source build gets from per-step `python <step>.py` subprocesses).
"""
import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--step":            # internal per-step dispatch (used by analyze_all)
        import importlib
        stem = argv[1]
        mod = importlib.import_module(stem)     # e.g. "embed_muq" — bundled as a top-level module
        return int(mod.main(argv[2:]) or 0)
    import analyze_all                           # default: run the whole ordered pipeline
    return int(analyze_all.main() or 0)


if __name__ == "__main__":
    sys.exit(main())
