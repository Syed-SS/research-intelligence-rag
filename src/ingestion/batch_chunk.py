import json
from pathlib import Path


PROCESSED_DIR = Path("data/processed")


def create_chunks(text: str) -> list[str]:
    chunks = []

    start = 0
    chunk_size = 1000
    chunk_overlap = 200

    while start < len(text):
        target_end = min(start + chunk_size, len(text))

        if target_end < len(text):
            end = text.rfind(" ", start, target_end)

            if end <= start:
                end = target_end
        else:
            end = target_end

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - chunk_overlap)

        boundary = text.find(" ", next_start, end)

        if boundary != -1:
            next_start = boundary + 1
        else:
            next_start = end

        start = next_start

    return chunks


def process_paper(input_path: Path) -> None:
    paper_id = input_path.stem
    output_path = PROCESSED_DIR / f"{paper_id}_chunks.json"

    text = input_path.read_text(encoding="utf-8")
    chunks = create_chunks(text)

    documents = []

    for index, chunk in enumerate(chunks, start=1):
        documents.append(
            {
                "chunk_id": f"{paper_id}-chunk-{index:04d}",
                "paper_id": paper_id,
                "text": chunk,
            }
        )

    output_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{paper_id}: {len(chunks)} chunks")


def main() -> None:
    text_files = sorted(PROCESSED_DIR.glob("RAG-*.txt"))

    if not text_files:
        print("No TXT files found.")
        return

    print(f"Found {len(text_files)} text files.")

    for input_path in text_files:
        process_paper(input_path)

    print("\nBatch chunking complete.")


if __name__ == "__main__":
    main()