import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    original = (root / "original.tex").read_text()
    revised_path = root.parent / "Stage2_RAW_Denoising_Research_Report_2026-10-23.tex"
    revised = revised_path.read_text()
    mapping = json.loads((root / "before.source_map.json").read_text())
    edits = json.loads((root / "edits.applied.json").read_text())
    original_lines = original.splitlines()
    revised_lines = revised.splitlines()
    changed_lines = [index for index, (before, after) in enumerate(zip(original_lines, revised_lines), 1) if before != after]
    expected_lines = sorted({edit["line"] for edit in edits})
    checks = {
        "line_count_unchanged": len(original_lines) == len(revised_lines),
        "changed_lines_match_edit_log": changed_lines == expected_lines,
        "only_prose_or_captions_changed": all(mapping["lines"][line - 1]["role"] in {"prose", "caption"} for line in changed_lines),
        "preamble_unchanged": original.split("\\begin{document}")[0] == revised.split("\\begin{document}")[0],
        "blank_lines_unchanged": [i for i, line in enumerate(original_lines) if not line.strip()] == [i for i, line in enumerate(revised_lines) if not line.strip()],
        "all_numeric_tokens_unchanged": re.findall(r"\d+(?:\.\d+)?", original) == re.findall(r"\d+(?:\.\d+)?", revised),
        "inline_math_unchanged": re.findall(r"\$[^$]*\$", original) == re.findall(r"\$[^$]*\$", revised),
        "latex_command_sequence_unchanged": re.findall(r"\\(?:[A-Za-z]+\*?|[^A-Za-z])", original) == re.findall(r"\\(?:[A-Za-z]+\*?|[^A-Za-z])", revised),
    }
    for command in ("section", "subsection", "subsubsection", "cite", "ref", "eqref", "label", "path", "url", "includegraphics", "texttt"):
        pattern = r"\\" + command + r"(?:\[[^\]]*\])?\{[^}]*\}"
        checks[f"{command}_arguments_unchanged"] = re.findall(pattern, original) == re.findall(pattern, revised)
    for environment in ("table", "longtable", "tabular", "equation", "equation*", "align", "verbatim", "thebibliography"):
        pattern = r"\\begin\{" + re.escape(environment) + r"\}.*?\\end\{" + re.escape(environment) + r"\}"
        checks[f"{environment}_blocks_unchanged"] = re.findall(pattern, original, re.S) == re.findall(pattern, revised, re.S)
    replay = original_lines.copy()
    for edit in edits:
        index = edit["line"] - 1
        assert replay[index].count(edit["old"]) == 1, edit
        replay[index] = replay[index].replace(edit["old"], edit["new"], 1)
    checks["edit_log_exactly_reproduces_revision"] = "\n".join(replay) + "\n" == revised
    result = {
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "revised_sha256": hashlib.sha256(revised.encode()).hexdigest(),
        "edits": len(edits),
        "changed_line_count": len(changed_lines),
        "changed_lines": changed_lines,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    (root / "integrity_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["all_passed"], "Revision changed protected content."


if __name__ == "__main__":
    main()
