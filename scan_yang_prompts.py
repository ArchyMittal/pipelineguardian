from pathlib import Path

from pipelineguardian.tools.notebook_parser import load_source
from pipelineguardian.tools.llm_only_auditor import _build_prompt

notebooks_dir = Path("eval/yang_corpus/notebooks")

valid = []
failed = []

for path in sorted(notebooks_dir.glob("*.ipynb")):
    try:
        parsed = load_source(str(path))
        prompt = _build_prompt(parsed)

        valid.append({
            "name": path.name,
            "source_chars": len(parsed.source),
            "prompt_chars": len(prompt),
            "whitespace_estimate": len(prompt.split()),
        })

    except Exception as exc:
        failed.append({
            "name": path.name,
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        })

valid.sort(key=lambda item: item["prompt_chars"], reverse=True)

print(f"NOTEBOOKS FOUND: {len(valid) + len(failed)}")
print(f"PROMPTS BUILT: {len(valid)}")
print(f"FAILED TO PARSE/BUILD: {len(failed)}")

print("\nTOP 20 LARGEST PROMPTS:")
for index, item in enumerate(valid[:20], start=1):
    print(
        f"{index:2}. {item['name']} | "
        f"source={item['source_chars']:,} chars | "
        f"prompt={item['prompt_chars']:,} chars | "
        f"whitespace estimate={item['whitespace_estimate']:,}"
    )

if failed:
    print("\nFAILED NOTEBOOKS:")
    for item in failed:
        print(f"- {item['name']} | {item['error']}")