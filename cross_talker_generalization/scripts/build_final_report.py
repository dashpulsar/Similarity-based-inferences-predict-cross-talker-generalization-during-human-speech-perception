from __future__ import annotations

import argparse

from ctg.final_report import build_final_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    build_final_report(arguments.repository, arguments.output)


if __name__ == "__main__":
    main()
