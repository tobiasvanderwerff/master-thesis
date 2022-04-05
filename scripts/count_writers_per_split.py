from pathlib import Path

here = Path(__file__).parent.resolve()
aachen_path = here / "../aachen_splits"

aachen_splits = {
    "train": (aachen_path / "train.uttlist").read_text().splitlines(),
    "validation": (aachen_path / "validation.uttlist").read_text().splitlines(),
    "test": (aachen_path / "test.uttlist").read_text().splitlines(),
}

d = Path("../iam_form_to_writerid.txt").read_text()
docid_to_writerid = dict([line.split() for line in d.splitlines()])

total_writers = 0
for split, docs in aachen_splits.items():
    print(split)

    writer_ids = set()
    for doc_id in docs:
        wid = docid_to_writerid[doc_id]
        writer_ids.add(wid)
    n_writers = len(writer_ids)
    total_writers += n_writers

    print(f"{n_writers} writers", end="\n\n")

print(f"Total: {total_writers}")
