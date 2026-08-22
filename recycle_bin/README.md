# Local recoverable archive

This directory stores superseded code, reports, intermediate tables, and other files that
were moved out of the production project without being permanently deleted.

Archive payloads are intentionally excluded from Git because they are historical,
duplicated elsewhere, or too large for ordinary source hosting. Only this policy document
is published. A local checkout used during the 2026-08-21 reorganization contains the
detailed archive indexes and file-level SHA-256 manifests.

Production code must never read from `recycle_bin/`. To inspect historical material, copy
only the required file or result group into a temporary review directory; do not restore
an entire legacy tree into the repository root.
