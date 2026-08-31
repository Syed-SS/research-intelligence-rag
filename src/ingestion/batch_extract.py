from pathlib import Path

from extract_text import extract_text


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    pdf_files = sorted(RAW_DIR.glob("RAG-*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        paper_id = pdf_path.stem
        output_path = PROCESSED_DIR / f"{paper_id}.txt"

        print(f"\nProcessing {paper_id}...")

        text = extract_text(pdf_path)

        output_path.write_text(text, encoding="utf-8")

        print(f"Extracted {len(text)} characters")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()