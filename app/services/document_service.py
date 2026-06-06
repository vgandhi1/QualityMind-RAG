"""
Document Processing Service
Handles parsing and chunking of various document formats (PDF, DOCX, CSV, JSON).

Now supports context-aware chunking with Docling for improved RAG quality.
"""

import logging
import re
from pathlib import Path
from typing import Any

import tiktoken
from unstructured.partition.auto import partition

logger = logging.getLogger("rag_app.document_service")


def parse_document(file_path: str) -> str:
    """
    Parse any document type and return extracted text.
    Uses fast direct read for simple text files (.txt, .md, .csv).
    Uses Unstructured.io for complex formats (PDF, DOCX, JSON, etc.).

    Args:
        file_path: Path to the document file

    Returns:
        str: Extracted text content from the document

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If parsing fails
    """
    # Verify file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Fast path for simple text files - bypass unstructured library
    # This is critical for Lambda performance (avoids 30+ second timeout)
    file_extension = Path(file_path).suffix.lower()
    if file_extension in [".txt", ".md", ".csv", ".log", ".json"]:
        try:
            logger.info(f"Using fast text read for {file_extension} file")
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Fast text read failed: {e}, falling back to unstructured")
        except Exception as e:
            logger.warning(f"Fast text read failed: {e}, falling back to unstructured")

    try:
        # Use Unstructured.io's auto partition for complex formats (PDF, DOCX, etc.)
        # strategy="fast" disables OCR (tesseract) for Lambda compatibility
        # OCR can be enabled by adding tesseract Lambda layer and using strategy="hi_res"
        logger.info(f"Using unstructured library for {file_extension} file")
        elements = partition(
            filename=file_path, strategy="fast"  # Fast mode: no OCR, works without tesseract
        )

        # Combine all elements into a single text string
        text = "\n\n".join([str(el) for el in elements])

        return text

    except Exception as e:
        raise Exception(f"Failed to parse document {file_path}: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",  # GPT-4 encoding
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        overlap: Number of overlapping tokens between chunks (default: 50)
        encoding_name: Tokenizer encoding to use (default: cl100k_base for GPT-4)

    Returns:
        List of dictionaries containing:
            - text: The chunk text
            - chunk_index: Index of the chunk
            - token_count: Number of tokens in the chunk
            - start_char: Starting character position
            - end_char: Ending character position
    """
    # Initialize tokenizer
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback to default encoding
        tokenizer = tiktoken.encoding_for_model("gpt-4")

    # Encode the entire text
    tokens = tokenizer.encode(text)

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Get chunk tokens
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)

        # Calculate character positions (approximate)
        if chunks:
            # For subsequent chunks, use the previous end position
            start_char = chunks[-1]["end_char"] - (overlap * 4)  # Rough estimate
            start_char = max(0, start_char)
        else:
            start_char = 0

        end_char = start_char + len(chunk_text)

        # Create chunk metadata
        chunk_data = {
            "text": chunk_text,
            "chunk_index": len(chunks),
            "token_count": len(chunk_tokens),
            "start_char": start_char,
            "end_char": end_char,
        }

        chunks.append(chunk_data)

        # Move to next chunk with overlap
        start_idx += chunk_size - overlap

        # Break if we've reached the end
        if end_idx >= len(tokens):
            break

    return chunks


def get_document_stats(file_path: str) -> dict[str, Any]:
    """
    Get statistics about a document.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary with document statistics
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse document
    text = parse_document(file_path)

    # Get token count
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(text)

    return {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_type": path.suffix,
        "character_count": len(text),
        "token_count": len(tokens),
        "estimated_chunks_512": (len(tokens) // 512) + 1,
    }


_PART_NUMBER_RE = re.compile(r"\b([A-Z]{2,8}-[A-Z0-9]{2,12}(?:-[A-Z0-9]{1,8})*)\b")
_RPN_RE = re.compile(r"\bRPN[:\s=]+(\d{1,3})\b", re.IGNORECASE)

_DOC_TYPE_SIGNALS: dict[str, list[str]] = {
    "pfmea": [
        "pfmea",
        "process fmea",
        "failure mode",
        "rpn",
        "occurrence",
        "detection rating",
        "severity rating",
    ],
    "dfmea": ["dfmea", "design fmea", "design failure"],
    "capa": ["corrective action", "preventive action", "capa", "root cause", "8d"],
    "8d": [
        "d1 team",
        "d2 problem",
        "d3 containment",
        "d4 root cause",
        "d5 permanent",
        "8-discipline",
    ],
    "control_plan": [
        "control plan",
        "control method",
        "reaction plan",
        "sample size",
        "sample frequency",
    ],
    "work_instruction": [
        "work instruction",
        "step-by-step",
        "operator instruction",
        "station instruction",
    ],
    "qms": ["quality manual", "iso 9001", "iatf 16949", "quality policy", "management review"],
    "spc": ["cpk", "cp index", "control chart", "ucl", "lcl", "western electric"],
}

_QUALITY_KEYWORDS = [
    "pfmea",
    "dfmea",
    "capa",
    "8d",
    "ncr",
    "control plan",
    "work instruction",
    "cpk",
    "cp ",
    "spc",
    "rma",
    "corrective action",
    "preventive action",
    "failure mode",
    "root cause",
    "containment",
    "gauge r&r",
    "ppap",
    "apqp",
    "severity",
    "occurrence",
    "detection",
    "rpn",
    "scrap",
    "rework",
    "disposition",
]


def extract_quality_metadata(text: str, filename: str) -> dict:
    """
    Detect document type, extract part numbers, RPN values, and quality keywords from chunk text.
    Results are stored as Pinecone metadata for filtered semantic search.
    """
    text_lower = text.lower()
    fn_lower = filename.lower()

    # Detect document type from filename first, then text content
    doc_type = "generic"
    for dtype, signals in _DOC_TYPE_SIGNALS.items():
        if any(s in fn_lower for s in signals) or any(s in text_lower for s in signals):
            doc_type = dtype
            break

    part_numbers = list({m.group(1) for m in _PART_NUMBER_RE.finditer(text)})[:10]
    rpn_values = [int(m.group(1)) for m in _RPN_RE.finditer(text)][:10]
    quality_keywords = [kw.strip() for kw in _QUALITY_KEYWORDS if kw in text_lower][:15]

    return {
        "doc_type": doc_type,
        "part_numbers": part_numbers,
        "rpn_values": rpn_values,
        "quality_keywords": quality_keywords,
    }


def parse_and_chunk_with_context(
    file_path: str, chunk_size: int = 512, min_chunk_size: int = 256
) -> list[dict[str, Any]]:
    """
    Parse and chunk document using Docling's context-aware approach.

    This is the RECOMMENDED method that provides:
    - Semantic boundary detection (no mid-sentence splits)
    - Hierarchical heading context preservation
    - Rich metadata (page numbers, captions, document structure)
    - Smart merging to ensure chunks are 256-512 tokens (not too small)

    Falls back to traditional token-based chunking if Docling is unavailable.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk (default: 512)
        min_chunk_size: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata
    """
    filename = Path(file_path).name
    file_extension = Path(file_path).suffix.lower()

    def _enrich(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach quality domain metadata to every chunk."""
        for chunk in chunks:
            meta = extract_quality_metadata(chunk["text"], filename)
            chunk.update(meta)
        return chunks

    # Fast path for simple text files — bypass Docling to avoid Lambda timeout
    if file_extension in [".txt", ".md", ".csv", ".log", ".json"]:
        logger.info(
            f"Using fast token-based chunking for {file_extension} file (bypassing Docling)"
        )
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)
        for chunk in chunks:
            chunk["headings"] = []
            chunk["page_numbers"] = []
            chunk["doc_items"] = []
            chunk["captions"] = []
        logger.info(f"Fast chunking complete: {len(chunks)} chunks")
        return _enrich(chunks)

    try:
        from app.services.docling_service import parse_and_chunk_document

        logger.info(f"Using Docling for context-aware chunking: {filename}")
        chunks = parse_and_chunk_document(
            file_path, chunk_size=chunk_size, min_chunk_size=min_chunk_size
        )
        logger.info(f"Docling chunking complete: {len(chunks)} chunks with heading context")
        return _enrich(chunks)

    except ImportError as e:
        logger.warning(f"Docling not available, falling back to token-based chunking: {e}")
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)
        for chunk in chunks:
            chunk["headings"] = []
            chunk["page_numbers"] = []
            chunk["doc_items"] = []
            chunk["captions"] = []
        logger.info(f"Token-based chunking complete: {len(chunks)} chunks (no context)")
        return _enrich(chunks)

    except Exception as e:
        logger.error(f"Docling failed, falling back to token-based chunking: {e}")
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)
        for chunk in chunks:
            chunk["headings"] = []
            chunk["page_numbers"] = []
            chunk["doc_items"] = []
            chunk["captions"] = []
        logger.warning(f"Using fallback chunking: {len(chunks)} chunks (no context)")
        return _enrich(chunks)
