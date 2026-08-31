import json
from pathlib import Path
from urllib.request import urlopen


METADATA_PATH = Path("data/metadata/papers.json")
RAW_DIR = Path("data/raw")


def download_papers() -> None:
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        paper_id = paper["paper_id"]
        arxiv_id = paper["arxiv_id"]

        output_path = RAW_DIR / f"{paper_id}.pdf"

        if output_path.exists():
            print(f"Already exists: {output_path}")
            continue

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        print(f"Downloading {paper_id} from arXiv...")

        with urlopen(pdf_url) as response:
            pdf_data = response.read()

        output_path.write_bytes(pdf_data)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    download_papers()