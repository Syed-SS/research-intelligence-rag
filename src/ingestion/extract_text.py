import sys
from pathlib import Path

from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src\\ingestion\\extract_text.py <paper_id>")
        return

    paper_id = sys.argv[1]

    pdf_path = RAW_DIR / f"{paper_id}.pdf"
    output_path = PROCESSED_DIR / f"{paper_id}.txt"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    text = extract_text(pdf_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(f"Extracted {len(text)} characters")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()