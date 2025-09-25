# Smart RAG Document Chunker (NLP-Solution)

Transform domain text files into retrieval-optimized hierarchical chunks for RAG systems. This repository includes a production-ready `SmartRAGChunker` that:

- Creates parent chunks (broader context) and child chunks (retrieval-optimized)
- Preserves semantic continuity via overlaps and sentence-aware splitting
- Outputs multiple formats for humans and machines

The default code and examples are tailored for a QoS Engineer domain, but the chunker works for any text content.

---

## Features

- Smart hierarchical chunking (section → parent → child)
- Context-window aware sizing with overlap for continuity
- Preserves code blocks and normalizes whitespace
- Adds helpful acronym expansions (configurable for telecom/QoS terms)
- Exports 4 formats: human-readable, structured JSON, RAG-ready TSV-like, and metadata

---

## Repository Structure

- `transform.py` – main implementation with `SmartRAGChunker` and helper entrypoints
- `output/` – example outputs produced by the chunker
- `QoS_Engineer_LLM.ipynb` – domain notebook (optional)
- `section 4.7.txt`, `section 5.3.txt` – example input sections
- `LICENSE` – project license

---

## Requirements

- Python 3.9+ (standard library only; no external packages required)
- Text files (`.txt`) as input

---

## Quick Start (Windows PowerShell)

1) Optionally create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Place your `.txt` source files in a folder of your choice. You can use this repo root or another directory.

3) Run the chunker:

```powershell
python .\transform.py | cat
```

By default, the script demonstrates processing using a specific path in the `__main__` block. See “Input folder” below to customize.

Outputs are written to an `output/` directory alongside your input folder. Example files include:

- `smart_rag_chunks_hierarchical.txt` (human-readable)
- `smart_rag_chunks_structured.json` (programmatic)
- `smart_rag_chunks_rag_ready.txt` (vector DB ingestion)
- `smart_rag_chunks_metadata.json` (stats and recommendations)

---

## Input folder

There are three common ways to set the input folder and context window:

- Edit the `__main__` section in `transform.py` and set your folder path:

```python
chunker = process_documents("F:/path/to/your/folder", 4096)
```

- Use the convenience function for the current directory:

```python
from transform import quick_process
quick_process()  # processes ./.txt files with 4096-token context window
```

- Call the main API with your own parameters (see next section).

Note: On Windows, prefer full paths like `F:/NLP-Solution/` or escape backslashes if using `C:\\...`.

---

## Programmatic Usage

```python
from transform import process_documents, SmartRAGChunker

# Process all .txt files in a folder and write outputs to that folder's ./output
chunker: SmartRAGChunker = process_documents(
    folder_path="./",        # folder containing .txt files
    context_window=4096       # model context window in tokens
)

if chunker:
    # Access all computed chunks
    all_chunks = chunker.all_chunks

    # Or access hierarchical view per section
    hierarchy = chunker.hierarchical_chunks
```

---

## Configuration

`SmartRAGChunker` exposes a `config` dictionary you can adjust after instantiation:

- `target_context_usage` – fraction of model context window to use for content (default 0.70)
- `available_context` – derived from `model_context_window * target_context_usage`
- `parent_chunk_size` – approximate characters per parent chunk (default 1200)
- `child_chunk_size` – approximate characters per child chunk (default 400)
- `overlap_size` – characters overlapped between sequential chunks (default 50)
- `min_chunk_size` – skip files shorter than this many characters (default 100)
- `sentence_boundary_preference` – prefer splitting on sentence ends (default True)
- `preserve_code_blocks` – protect fenced ``` code blocks (default True)

Domain acronyms used for first-mention expansions are defined in `domain_acronyms`. You can extend or modify this map to fit your domain.

---

## Outputs

All outputs are created under `<input_folder>/output/`:

- `smart_rag_chunks_hierarchical.txt` – human-friendly overview of sections, parent/child chunks, and previews
- `smart_rag_chunks_structured.json` – machine-friendly structure of all chunks and metadata
- `smart_rag_chunks_rag_ready.txt` – line format: `ID|LEVEL|SECTION|PARENT|TOKENS|HASH|CONTENT` (with safe escaping)
- `smart_rag_chunks_metadata.json` – statistics, per-section breakdown, and optimization recommendations

---

## API Surface (selected)

- `SmartRAGChunker.load_sections(file_pattern="*.txt", folder_path="./") -> bool`
- `SmartRAGChunker.create_hierarchical_chunks() -> List[Dict]`
- `SmartRAGChunker.save_chunked_output(base_filename="smart_rag_chunks") -> Dict[str, str]`
- `process_documents(folder_path="./", context_window=4096) -> SmartRAGChunker | None`
- `quick_process()` and `process_large_context()` helpers

---

## Tips & Troubleshooting

- Use UTF-8 text files. Extremely short files (<100 chars) are skipped by default.
- For large contexts (e.g., 128k-token models), call `process_large_context()` or pass a larger `context_window`.
- If you prefer outputs in this repo’s root `output/` folder, place your `.txt` files here and call `quick_process()`.
- On Windows PowerShell, append `| cat` to long-running commands if a pager is invoked.

---

## License

See `LICENSE` for details.

