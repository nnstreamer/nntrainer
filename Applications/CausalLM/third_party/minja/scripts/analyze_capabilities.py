#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List


def generate_markdown_table(files_data: List[tuple[str, Dict]]) -> str:
    """Generate a markdown table from the capabilities data."""
    if not files_data:
        return "No capability files found."
    
    all_caps = set()
    for _, data in files_data:
        all_caps.update(data.keys())
    all_caps = sorted(all_caps)
    
    lines = [
        "| Model | " + " | ".join(c.replace('_', ' ') for c in all_caps) + " |",
        "|" + "|".join("-" * (len(cap) + 2) for cap in ["Model"] + list(all_caps)) + "|",
    ]
    
    # Sort data by most supports and least requires
    def sort_key(item):
        model, data = item
        supports_count = sum(1 for k, v in data.items() 
                           if k.startswith("supports_") and str(v).lower() == "true")
        requires_count = sum(1 for k, v in data.items() 
                           if k.startswith("requires_") and str(v).lower() == "true")
        return (-supports_count, requires_count)  # negative for descending supports
    
    for model, data in sorted(files_data, key=sort_key):
        model_name = os.path.basename(model).replace(".caps.json", "")
        row = [model_name]
        for cap in all_caps:
            raw_value = str(data.get(cap, "N/A")).lower()
            if raw_value == "true":
                if cap.startswith("supports_"):
                    value = "✅"
                elif cap.startswith("requires_"):
                    value = "⚠️"
                else:
                    value = raw_value
            elif raw_value == "false":
                value = ""
            else:
                value = raw_value
            row.append(value)
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)

def main():
    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"
    
    files_data = [
        (str(f), json.loads(f.read_text()))
        for f in list((build_dir / "tests").rglob("*.caps.json"))
    ]
    
    markdown = generate_markdown_table(files_data)
    
    (build_dir / "capabilities.md").write_text(markdown)
    
    print(markdown)

if __name__ == "__main__":
    main()
