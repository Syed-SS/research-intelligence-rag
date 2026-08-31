import json
from pathlib import Path


INPUT_PATH = Path("data/processed/RAG-001.txt")
OUTPUT_PATH = Path("data/processed/RAG-001_chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_chunks(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a whitespace boundary.
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

        # Start the next chunk around the desired overlap.
        next_start = max(0, end - CHUNK_OVERLAP)

        # Move the overlap start forward to a whitespace boundary.
        boundary = text.find(" ", next_start, end)

        if boundary != -1:
            next_start = boundary + 1
        else:
            next_start = end

        start = next_start

    return chunks


def main() -> None:
    text = INPUT_PATH.read_text(encoding="utf-8")

    chunks = create_chunks(text)

    documents = []

    for index, chunk in enumerate(chunks, start=1):
        documents.append(
            {
                "chunk_id": f"RAG-001-chunk-{index:04d}",
                "paper_id": "RAG-001",
                "text": chunk,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created {len(documents)} chunks")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()