# tools/gen_yaml.py
import re, pathlib, collections, yaml

PKG_RE = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z0-9_]+)')
builtins = {"os", "sys", "re", "math", "glob", "argparse", "copy", "pathlib", "logging"}

def find_imports(root="."):
    pkgs = collections.Counter()
    for path in pathlib.Path(root).rglob("*.py"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = PKG_RE.match(line)
            if m:
                pkgs[m.group(1).split(".")[0]] += 1
    return sorted(set(pkgs) - builtins)

deps = find_imports()
yaml_dict = {
    "name": "sdt",
    "channels": ["conda-forge"],
    "dependencies": ["python=3.8", "pip", {"pip": deps}],
}

print(yaml.dump(yaml_dict, sort_keys=False))
