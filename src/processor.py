import io
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.config import Config

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

import base64


def _image_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class PDFProcessor:
    """
    Processes PDFs (native and scanned) and produces text chunks that include
    page-level bounding boxes and a page image for precise citation highlighting.

    Strategy:
    - Use PyMuPDF (fitz) to extract page text blocks and their bboxes for native PDFs.
    - If a page has little or no extracted text, render the page to an image and run
      OCR (pytesseract) to obtain word bounding boxes.
    - Chunk text by concatenating contiguous text elements until chunk_size is reached.
    - Store per-chunk metadata: source filename, page number, list of bboxes (in image coords),
      and the page image bytes for UI highlighting.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        self.tesseract_available = False
        if pytesseract is not None:
            try:
                _ = pytesseract.get_tesseract_version()
                self.tesseract_available = True
            except Exception:
                self.tesseract_available = False

    def _render_page_image(self, page, zoom: int = 2):
        """Render a PyMuPDF page to a PIL Image."""
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        mode = "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return img

    def _extract_native_blocks(self, page, zoom: int = 2) -> List[Dict[str, Any]]:
        """Extract text blocks and bbox from a PyMuPDF page.

        Returns list of elements with keys: text, bbox (x0,y0,x1,y1) in page image coords.
        """
        blocks = page.get_text("blocks")
        elements: List[Dict[str, Any]] = []
        for b in blocks:
            x0, y0, x1, y1, text, block_no = b[0], b[1], b[2], b[3], b[4], b[5]
            text = text.strip()
            if not text:
                continue
            sx0, sy0, sx1, sy1 = x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom
            elements.append({"text": text, "bbox": (sx0, sy0, sx1, sy1)})
        return elements

    def _extract_ocr_elements(self, image) -> List[Dict[str, Any]]:
        """Run Tesseract OCR and return elements with text and bbox in image coords."""
        if pytesseract is None:
            return []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as e:
            raise RuntimeError(str(e))

        elements: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            elements.append({"text": text, "bbox": (x, y, x + w, y + h)})
        return elements

    def _chunk_elements(self, elements: List[Dict[str, Any]]) -> List[Tuple[str, List[Tuple[float, float, float, float]]]]:
        """Group contiguous elements into text chunks and collect bboxes.

        Returns list of (chunk_text, [bbox, ...])
        """
        chunks: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []
        current_text = []
        current_bboxes: List[Tuple[float, float, float, float]] = []
        for el in elements:
            txt = el["text"]
            bbox = el["bbox"]
            # If adding this text would exceed chunk size, flush current chunk
            if current_text and (len(" ".join(current_text)) + len(txt) > Config.CHUNK_SIZE):
                chunks.append((" ".join(current_text), current_bboxes))
                current_text = []
                current_bboxes = []
            current_text.append(txt)
            current_bboxes.append(bbox)

        if current_text:
            chunks.append((" ".join(current_text), current_bboxes))
        return chunks

    def process_document(self, pdf_file) -> List[Document]:
        """Process a PDF file-like object and return LangChain Documents with metadata.

        pdf_file is expected to be a Streamlit UploadedFile (has .name and .read()).
        """
        try:
            raw = pdf_file.read()
            if fitz is None:
                raise RuntimeError("PyMuPDF (fitz) is required for PDF processing.")

            doc = fitz.open(stream=raw, filetype="pdf")
            all_documents: List[Document] = []

            for page_number in range(len(doc)):
                page = doc[page_number]

                if Image is None:
                    raise RuntimeError("Pillow is required for rendering page images.")
                img = self._render_page_image(page, zoom=2)
                img_bytes = _image_to_bytes(img)
                img_w, img_h = img.width, img.height

                elements = self._extract_native_blocks(page)

                total_chars = sum(len(el["text"]) for el in elements)
                if total_chars < 50 and pytesseract is not None:
                    if not self.tesseract_available:
                        print(
                            "Tesseract OCR is not available (binary missing or not in PATH). "
                            "Scanned pages will not be OCR'd. See README for installation instructions."
                        )
                    else:
                        try:
                            ocr_elements = self._extract_ocr_elements(img)
                            elements = ocr_elements
                        except Exception as ocr_e:
                            print(f"Tesseract OCR failed on page {page_number+1}: {ocr_e}")

                if not elements:
                    continue

                chunks = self._chunk_elements(elements)

                for idx, (chunk_text, bboxes) in enumerate(chunks):
                    metadata = {
                        "source": pdf_file.name,
                        "page": page_number + 1,
                        "chunk_id": idx,
                        "bboxes": bboxes,
                        "page_image": base64.b64encode(img_bytes).decode("utf-8"),
                        "page_image_width": img_w,
                        "page_image_height": img_h,
                    }
                    all_documents.append(Document(page_content=chunk_text, metadata=metadata))

            return all_documents
        except Exception as e:
            print(f"Error processing document {getattr(pdf_file, 'name', '<memory>')}: {str(e)}")
            return []