"""Fix all non-ASCII characters in pipeline.py so it runs on Windows cp1252."""
import pathlib, re

p = pathlib.Path(__file__).parent / "pipeline.py"
content = p.read_text(encoding="utf-8")

# Replace every non-ASCII character with a plain ASCII equivalent
# Box-drawing / separator lines in comments -> dashes
content = re.sub(r"[^\x00-\x7F]", "-", content)

p.write_text(content, encoding="utf-8")
print("Done: pipeline.py re-written as pure ASCII")
