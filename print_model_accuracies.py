from __future__ import annotations

import json
from pathlib import Path


ACCURACY_DIR = Path("model_accuracies")


def main() -> None:
    json_files = sorted(ACCURACY_DIR.glob("*_accuracies.json"))

    if not json_files:
        raise FileNotFoundError(f"No accuracy files found in {ACCURACY_DIR}")

    model_entries = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        test_f1_macro = data.get("test_f1_macro", "0")
        if isinstance(test_f1_macro, str):
            test_f1_macro = test_f1_macro.rstrip("%")

        try:
            score = float(test_f1_macro)
        except (TypeError, ValueError):
            score = float("-inf")

        model_entries.append((score, json_file, data))

    for index, (_, json_file, data) in enumerate(sorted(model_entries, key=lambda item: item[0], reverse=True)):

        model_name = data.get("model_name", json_file.stem.replace("_accuracies", ""))
        print(f"Model: {model_name}")
        print(json.dumps(data, indent=4, ensure_ascii=False))

        if index < len(json_files) - 1:
            print()


if __name__ == "__main__":
    main()