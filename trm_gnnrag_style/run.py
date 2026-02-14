"""Backward-compatible entrypoint.

Use `python -m trm_rag_style.run` for the TRM-first package name.
"""

from trm_rag_style.run import main

if __name__ == '__main__':
    main()
