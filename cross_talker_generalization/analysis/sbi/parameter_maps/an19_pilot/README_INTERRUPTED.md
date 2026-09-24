# Interrupted AN19 batch — exclude from analysis

Stopped on September 18 when Zhengyang requested X21 priority. Only this run's Python process and its R subprocess descendants were terminated; partial outputs were preserved. The pre-change source files are retained in `source_snapshot/`.

This unfinished batch also used expm1 across the full k range. That calculation can lose small exponential differences at large k; the replacement runner uses a stable, affine-equivalent shifted exponential with expm1 restricted to small exponent spans. Do not use these partial AN19 scores as a validated landscape or merge them into the X21 run.
