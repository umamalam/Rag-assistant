"""
ingest.py
Indexes everything in documents/ (PDF, Markdown, TXT, DOCX, EML, MBOX),
including subfolders, into a local ChromaDB store using local embeddings.

  python3 ingest.py                 index new/changed files (safe to re-run)
  python3 ingest.py --remove NAME   delete one file's chunks from the index
  python3 ingest.py --wipe          delete the entire index

Files are tracked by content hash: unchanged files are skipped, edited files
are re-indexed, so the index never holds stale text.
"""

import argparse
import hashlib
import os

import loaders
import store

DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "../documents")
CHUNK_SIZE = 800      # characters per chunk (~150-200 words)
CHUNK_OVERLAP = 150   # overlap so ideas aren't cut at chunk boundaries
BATCH = 256           # Chroma rejects very large single add() calls


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding window over characters, preferring to end on a sentence."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind(". ", start, end)
            if cut - start > chunk_size * 0.5:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 30:  # drop tiny fragments
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_files(root):
    found = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if os.path.splitext(name)[1].lower() in loaders.SUPPORTED:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def ingest_file(collection, root, path):
    rel = os.path.relpath(path, root)
    digest = file_hash(path)

    existing = collection.get(where={"source": rel}, limit=1, include=["metadatas"])
    if existing["ids"]:
        if existing["metadatas"][0].get("content_hash") == digest:
            print(f"  Skipping {rel} (unchanged)")
            return 0
        print(f"  {rel} changed -- re-indexing")
        collection.delete(where={"source": rel})

    print(f"  Processing {rel}...")
    units = loaders.load_file(path)
    file_ts = os.path.getmtime(path)
    base = hashlib.sha1(rel.encode()).hexdigest()[:12]
    ext = os.path.splitext(path)[1].lower().lstrip(".")

    ids, docs, metas = [], [], []
    for u in units:
        for i, chunk in enumerate(chunk_text(u["text"])):
            ids.append(f"{base}::{u['page']}::{i}")
            docs.append(chunk)
            metas.append({
                "source": rel,
                "page": u["page"],
                "location": u["location"],
                "type": ext,
                "content_hash": digest,
                "doc_ts": float(u.get("doc_ts") or file_ts),
            })

    if not docs:
        print(f"  WARNING: no extractable text in {rel} (scanned/image PDF?)")
        return 0

    for i in range(0, len(docs), BATCH):
        sl = slice(i, i + BATCH)
        collection.add(
            ids=ids[sl], documents=docs[sl], metadatas=metas[sl],
            embeddings=store.embed(docs[sl]),
        )
    print(f"  Done: {len(docs)} chunks from {len(units)} units")
    return len(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", metavar="NAME", help="delete one file from the index")
    ap.add_argument("--wipe", action="store_true", help="delete the whole index")
    args = ap.parse_args()

    if args.wipe:
        store.get_client().delete_collection(store.COLLECTION_NAME)
        print("Index wiped. (Your original files in documents/ are untouched.)")
        return

    collection = store.get_collection()

    if args.remove:
        collection.delete(where={"source": args.remove})
        print(f"Removed '{args.remove}' from the index.")
        return

    if not os.path.isdir(DOCUMENTS_DIR):
        print(f"ERROR: documents folder not found at {DOCUMENTS_DIR}")
        return
    files = find_files(DOCUMENTS_DIR)
    if not files:
        print(f"No supported files in {DOCUMENTS_DIR} "
              f"({', '.join(sorted(loaders.SUPPORTED))})")
        return

    print(f"Found {len(files)} file(s) in {DOCUMENTS_DIR}\n")
    added = sum(ingest_file(collection, DOCUMENTS_DIR, p) for p in files)
    print(f"\nDone. Index now has {collection.count()} chunks ({added} added this run).")


if __name__ == "__main__":
    main()
