import importlib.util
import json
import re
import sys
from pathlib import Path


def plain_text(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\$[^$]*\$", " [MATH] ", text)
    text = re.sub(r"\\(?:path|url)\{[^}]*\}", " [PATH] ", text)
    text = re.sub(r"\\(?:eqref|ref|cite)\{[^}]*\}", " [REF] ", text)
    text = re.sub(r"\\(?:label)\{[^}]*\}", "", text)
    for command, value in {"dn": "DN", "db": "dB", "psnr": "PSNR", "rmse": "RMSE"}.items():
        text = re.sub(r"\\" + command + r"\b(?:\{\})?", value, text)
    text = re.sub(r"\\([%&_#])", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    return re.sub(r"[{}]", "", text).strip()


def project(source: Path, destination: Path) -> dict:
    projected = []
    mapping = []
    inside_document = False
    blocked = []
    section = 0
    for number, line in enumerate(source.read_text().splitlines(), 1):
        role = "layout"
        rendered = ""
        if "\\begin{document}" in line:
            inside_document = True
        if inside_document:
            openings = re.findall(r"\\begin\{([^}]+)\}", line)
            closings = re.findall(r"\\end\{([^}]+)\}", line)
            for environment in openings:
                if environment in {"table", "tabular", "longtable", "equation", "equation*", "align", "align*", "verbatim", "thebibliography"}:
                    blocked.append(environment)
            heading = re.match(r"\\(section|subsection|subsubsection)\*?\{(.*)\}\s*$", line)
            caption_line = re.sub(r"\\label\{[^}]*\}", "", line)
            caption = re.match(r"\\caption(?:\[[^\]]*\])?\{(.*)\}", caption_line)
            if heading:
                section += 1
                role = "heading"
                rendered = "#" * {"section": 1, "subsection": 2, "subsubsection": 3}[heading[1]] + " " + plain_text(heading[2])
            elif caption:
                role = "caption"
                rendered = plain_text(caption[1])
            elif blocked:
                role = "protected_environment"
            elif re.search(r"[\u4e00-\u9fff]", line) and not line.startswith("\\"):
                role = "prose"
                rendered = plain_text(line)
            elif re.match(r"\\(?:textbf|emph)\{", line):
                role = "prose"
                rendered = plain_text(line)
            for environment in closings:
                if environment in blocked:
                    blocked.remove(environment)
        projected.append(rendered)
        mapping.append({"line": number, "role": role, "section": section, "text": rendered})
    text = "\n".join(projected) + "\n"
    destination.write_text(text)
    return {"source": str(source), "projection": str(destination), "hanzi": len(re.findall(r"[一-鿿]", text)), "lines": mapping}


def main() -> None:
    root = Path(__file__).resolve().parent
    phase = sys.argv[1]
    source = root / "original.tex" if phase == "before" else root.parent / "Stage2_RAW_Denoising_Research_Report_2026-10-23.tex"
    projection = root / f"{phase}.txt"
    mapping = project(source, projection)
    (root / f"{phase}.source_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    skill = Path("/home/jingzeliang/.codex/skills/lieflat-less-ai-tone/scripts/detect.py")
    spec = importlib.util.spec_from_file_location("tone_detect", skill)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    detector.CAND_CAPS = {key: 1000000 for key in detector.CAND_CAPS}
    documents = [(projection.name, projection.read_text())]
    candidates = detector.extract_candidates(documents)
    for kind in ("rename", "cohesion", "colon", "iso3"):
        for index, candidate in enumerate(candidates[kind], 1):
            candidate["id"] = f"{kind}-{index:03}"
            candidate["source_role"] = mapping["lines"][candidate["line"] - 1]["role"]
            candidate["section"] = mapping["lines"][candidate["line"] - 1]["section"]
    (root / f"{phase}.candidates.full.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    findings = detector.measure_docs(documents)
    for name, hits in findings["hits"].items():
        for index, hit in enumerate(hits, 1):
            hit["id"] = f"{name}-{index:03}"
            hit["source_role"] = mapping["lines"][hit["line"] - 1]["role"]
            hit["source_text"] = mapping["lines"][hit["line"] - 1]["text"]
    punctuation = []
    for item in mapping["lines"]:
        for name, mark in (("破折号", "——"), ("分号", "；"), ("引号", "“")):
            for match in re.finditer(mark, item["text"]):
                punctuation.append({"name": name, "line": item["line"], "offset": match.start(), "source_role": item["role"], "source_text": item["text"]})
    (root / f"{phase}.hits.full.json").write_text(json.dumps({"regex": findings["hits"], "syn_hits": findings["syn_hits"], "punctuation": punctuation}, ensure_ascii=False, indent=2))
    print(json.dumps({"phase": phase, "hanzi": mapping["hanzi"], "meta": findings["meta"], "candidates": candidates["meta"], "hits": {key: len(value) for key, value in findings["hits"].items()}, "punctuation": {name: sum(x["name"] == name for x in punctuation) for name in ("破折号", "分号", "引号")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
