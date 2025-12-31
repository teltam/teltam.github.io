#!/usr/bin/env python3
"""Generate index.md from markdown files in md/ directory."""

from datetime import datetime
from pathlib import Path

MD_DIR = Path("md")
INDEX_FILE = Path("index.md")

def extract_title(filepath):
    """Extract title from second line of file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return filepath.stem
    return lines[1].split(':')[1].strip().strip("'").strip()

def generate_index():
    """Generate index.md with list of posts."""
    posts = []

    for md_file in MD_DIR.glob("*.md"):
        title = extract_title(md_file)
        date = datetime.fromtimestamp(md_file.stat().st_birthtime)
        html_file = f"posts/{md_file.stem}.html"
        posts.append((date, title, html_file))

    # Sort by date, newest first
    posts.sort(key=lambda x: x[0], reverse=True)

    lines = [
        "---",
        "title: Hello world!",
        "---",
        "",
        "## Posts",
        "",
    ]

    for date, title, html_file in posts:
        date_str = date.strftime("%b %d, %Y")
        lines.append(f"- [{title} ({date_str})]({html_file})")

    lines.append("")
    INDEX_FILE.write_text("\n".join(lines))

if __name__ == "__main__":
    generate_index()
