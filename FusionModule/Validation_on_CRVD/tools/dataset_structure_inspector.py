from pathlib import Path
from datetime import datetime

# ====== 改这里：你要检测的根目录 ======
ROOT = Path(r"E:\CRVD_dataset")  # 或你自己的路径
# ====================================

IGNORE_DIRS = {".git", ".idea", "__pycache__", ".venv", "venv", "node_modules"}
MAX_DEPTH = 20  # 防止目录太深刷屏

def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def walk_tree(root: Path, max_depth: int = 20):
    root = root.resolve()
    lines = []
    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"不是文件夹: {root}")

    def _walk(dir_path: Path, depth: int):
        if depth > max_depth:
            lines.append("  " * depth + "…(max_depth reached)")
            return

        # 先列目录，再列文件；名字排序
        entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for p in entries:
            if p.is_dir() and p.name in IGNORE_DIRS:
                lines.append("  " * depth + f"[D] {p.name}/  (ignored)")
                continue

            if p.is_dir():
                lines.append("  " * depth + f"[D] {p.name}/")
                _walk(p, depth + 1)
            else:
                try:
                    st = p.stat()
                    size_kb = st.st_size / 1024
                    mtime = fmt_time(st.st_mtime)
                    lines.append("  " * depth + f"[F] {p.name}  ({size_kb:.1f} KB, {mtime})")
                except Exception as e:
                    lines.append("  " * depth + f"[F] {p.name}  (stat failed: {e})")

    lines.append(f"ROOT: {root}")
    _walk(root, 0)
    return "\n".join(lines)

if __name__ == "__main__":
    out = walk_tree(ROOT, MAX_DEPTH)
    print(out)

    # 同时保存一份到根目录
    save_path = ROOT / "_file_tree.txt"
    try:
        save_path.write_text(out, encoding="utf-8")
        print(f"\n已保存: {save_path}")
    except Exception as e:
        print(f"\n保存失败: {e}")
