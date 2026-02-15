"""
File extraction utilities for API file upload endpoints.
Reused and adapted from app.py for FastAPI UploadFile handling.
"""

import io
from fastapi import UploadFile, HTTPException


async def extract_text_from_file(file: UploadFile) -> str:
    """
    Extract text from uploaded file (TXT, PDF, DOCX).

    Args:
        file: FastAPI UploadFile object

    Returns:
        Extracted text content as string

    Raises:
        HTTPException: If file type is unsupported or extraction fails
    """
    filename = file.filename.lower() if file.filename else ""

    # Read file content
    content = await file.read()

    if filename.endswith(".txt"):
        try:
            return content.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode TXT file: {str(e)}"
            )

    elif filename.endswith(".pdf"):
        try:
            import pdfplumber
            pdf_bytes = io.BytesIO(content)
            with pdfplumber.open(pdf_bytes) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n\n".join(pages)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF support requires 'pdfplumber'. Install: pip install pdfplumber"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )

    elif filename.endswith(".docx"):
        try:
            from docx import Document
            doc_bytes = io.BytesIO(content)
            doc = Document(doc_bytes)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="DOCX support requires 'python-docx'. Install: pip install python-docx"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from DOCX: {str(e)}"
            )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {filename}. Supported: .txt, .pdf, .docx"
        )


def validate_file_size(file_size: int, max_size_mb: int = 10) -> None:
    """
    Validate uploaded file size.

    Args:
        file_size: File size in bytes
        max_size_mb: Maximum allowed size in MB

    Raises:
        HTTPException: If file exceeds size limit
    """
    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        )
