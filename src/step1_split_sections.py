"""
Step 1: Split main PDF into 15 horizontal sections
Output: section_01.pdf, section_02.pdf, ..., section_15.pdf
"""

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
from datetime import datetime


class SectionSplitter:
    """Splits a PDF page into 15 horizontal sections."""
    
    def __init__(self, pdf_path: str, base_dir: str = "."):
        self.pdf_path = Path(pdf_path)
        self.base_dir = Path(base_dir)
        
        # Setup directories
        self.data_dir = self.base_dir / "data"
        self.input_dir = self.data_dir / "input"
        self.sections_dir = self.data_dir / "sections"
        self.output_dir = self.base_dir / "output"
        
        # Create directories
        for dir_path in [self.input_dir, self.sections_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Open PDF with both libraries
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.plumber_doc = pdfplumber.open(str(self.pdf_path))
        
        self.log_entries = []
    
    def log(self, message: str):
        """Log a message to console and store for file output."""
        print(message)
        self.log_entries.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def detect_top_section_boundary(self, page_num: int) -> float:
        """Detect Y-coordinate where top section (heading/description) ends."""
        page = self.plumber_doc.pages[page_num]
        page_height = page.height
        
        top_portion_height = page_height * 0.15
        words = page.extract_words()
        
        top_words = [w for w in words if w['top'] < top_portion_height]
        
        if not top_words:
            return min(page_height * 0.1, 100)
        
        top_words_sorted = sorted(top_words, key=lambda w: w['top'])
        
        font_sizes = [w.get('size', 10) for w in top_words_sorted if 'size' in w]
        if font_sizes:
            max_font_size = max(font_sizes)
            heading_words = [w for w in top_words_sorted if w.get('size', 10) >= max_font_size * 0.9]
            if heading_words:
                heading_bottom = max(w['bottom'] for w in heading_words)
            else:
                heading_bottom = max(w['bottom'] for w in top_words_sorted[:5])
        else:
            heading_bottom = max(w['bottom'] for w in top_words_sorted[:5])
        
        all_words_below_heading = [w for w in words if w['top'] > heading_bottom]
        if all_words_below_heading:
            all_words_below_heading.sort(key=lambda w: w['top'])
            
            for i in range(len(all_words_below_heading) - 1):
                gap = all_words_below_heading[i + 1]['top'] - all_words_below_heading[i]['bottom']
                if gap > 30:
                    next_word_text = all_words_below_heading[i + 1]['text'].strip()
                    section_keywords = ['Instrument', 'Valves', 'Pumps', 'Vessels', 'Filters', 
                                      'Compressors', 'Heat', 'Dryers', 'Mixers', 'Crushers',
                                      'Centrifuges', 'Motors', 'Peripheral', 'Piping']
                    if any(keyword.lower() in next_word_text.lower() for keyword in section_keywords):
                        return all_words_below_heading[i]['bottom']
            
            return min(heading_bottom + 100, page_height * 0.15)
        else:
            return min(heading_bottom + 50, page_height * 0.1)
    
    def detect_horizontal_dotted_lines(self, page_img: Image.Image, page_start_y: float, 
                                       scale_y: float, page_width: float) -> List[float]:
        """Detect horizontal dashed/dotted lines on the page."""
        img_array = np.array(page_img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        page_start_y_img = int(page_start_y * scale_y)
        gray_cropped = gray[page_start_y_img:, :]
        
        edges = cv2.Canny(gray_cropped, 50, 150, apertureSize=3)
        
        kernel_horizontal = np.ones((1, 15), np.uint8)
        dilated = cv2.dilate(edges, kernel_horizontal, iterations=1)
        dilated = cv2.erode(dilated, kernel_horizontal, iterations=1)
        
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100,
                               minLineLength=int(page_width * scale_y * 0.5),
                               maxLineGap=20)
        
        horizontal_lines_y = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 3 or angle > 177:
                    y_avg = (y1 + y2) / 2
                    y_pdf = (y_avg + page_start_y_img) / scale_y
                    horizontal_lines_y.append(y_pdf)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > page_width * scale_y * 0.5 and h < 5:
                y_center = y + h / 2
                y_pdf = (y_center + page_start_y_img) / scale_y
                
                if not any(abs(existing - y_pdf) < 5 for existing in horizontal_lines_y):
                    horizontal_lines_y.append(y_pdf)
        
        horizontal_lines_y = sorted(set(horizontal_lines_y))
        
        grouped_lines = []
        for line_y in horizontal_lines_y:
            if not grouped_lines or abs(line_y - grouped_lines[-1]) > 3:
                grouped_lines.append(line_y)
            else:
                grouped_lines[-1] = (grouped_lines[-1] + line_y) / 2
        
        return grouped_lines
    
    def detect_section_boundaries(self, page_num: int, page_start_y: float) -> List[Tuple[float, float]]:
        """Detect 15 section boundaries starting from page_start_y using horizontal dashed/dotted lines."""
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        scale_y = pix.height / page.rect.height
        scale_x = pix.width / page.rect.width
        
        self.log(f"    Detecting horizontal dashed/dotted lines...")
        horizontal_lines = self.detect_horizontal_dotted_lines(page_img, page_start_y, scale_y, page_width)
        self.log(f"    Found {len(horizontal_lines)} horizontal lines (expected ~30)")
        
        boundaries = []
        
        if len(horizontal_lines) >= 2:
            for i in range(0, len(horizontal_lines) - 1, 2):
                y_start = horizontal_lines[i]
                y_end = horizontal_lines[i + 1] if i + 1 < len(horizontal_lines) else page_height * 0.98
                boundaries.append((y_start, y_end))
            
            if len(horizontal_lines) % 2 == 1:
                y_start = horizontal_lines[-2]
                y_end = page_height * 0.98
                if boundaries:
                    boundaries[-1] = (y_start, y_end)
        else:
            # Fallback: divide evenly
            section_height = (page_height - page_start_y) / 15
            for i in range(15):
                y_start = page_start_y + i * section_height
                y_end = page_start_y + (i + 1) * section_height
                boundaries.append((y_start, y_end))
        
        # Ensure we have exactly 15 sections
        if len(boundaries) > 15:
            boundaries = boundaries[:15]
        elif len(boundaries) < 15:
            while len(boundaries) < 15 and boundaries:
                largest_idx = max(range(len(boundaries)), 
                                 key=lambda i: boundaries[i][1] - boundaries[i][0])
                y_start, y_end = boundaries[largest_idx]
                y_mid = (y_start + y_end) / 2
                boundaries[largest_idx] = (y_start, y_mid)
                boundaries.insert(largest_idx + 1, (y_mid, y_end))
        
        return boundaries[:15]
    
    def save_section_as_pdf(self, page_num: int, section_idx: int, 
                           y_start: float, y_end: float) -> Path:
        """Save a section as a separate PDF file with naming: section_XX.pdf"""
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        section_height = y_end - y_start
        
        # Define the source rectangle (clip region) on the original page
        source_rect = fitz.Rect(0, y_start, page_width, y_end)
        
        # Render the section region as an image
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=source_rect)
        
        # Create a new PDF document
        section_doc = fitz.open()
        
        # Create a new page with the section dimensions
        section_page = section_doc.new_page(width=page_width, height=section_height)
        
        # Calculate the position to insert the image
        img_width_pt = pix.width / zoom
        img_height_pt = pix.height / zoom
        
        # Insert the image into the page
        img_rect = fitz.Rect(0, 0, img_width_pt, img_height_pt)
        section_page.insert_image(img_rect, pixmap=pix)
        
        # Filename: section_01.pdf, section_02.pdf, etc.
        filename = f"section_{section_idx+1:02d}.pdf"
        filepath = self.sections_dir / filename
        
        # Save the section PDF
        section_doc.save(str(filepath))
        section_doc.close()
        pix = None  # Free memory
        
        return filepath
    
    def split_pdf_into_sections(self, page_num: int = 0) -> List[Path]:
        """Split PDF page into 15 sections and save them."""
        self.log("="*60)
        self.log("STEP 1: Section Splitting")
        self.log("="*60)
        self.log(f"Splitting page {page_num + 1} into sections...")
        
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Detect top section boundary
        self.log("  Detecting top section boundary...")
        page_start_y = self.detect_top_section_boundary(page_num)
        self.log(f"  Top section ends at Y={page_start_y:.1f}")
        
        # Detect section boundaries
        self.log("  Detecting section boundaries...")
        section_boundaries = self.detect_section_boundaries(page_num, page_start_y)
        self.log(f"  Found {len(section_boundaries)} sections")
        
        # Save each section as PDF
        section_pdfs = []
        for section_idx, (y_start, y_end) in enumerate(section_boundaries[:15]):
            self.log(f"  Saving section {section_idx + 1}/15 (Y: {y_start:.1f}-{y_end:.1f})...")
            
            section_pdf_path = self.save_section_as_pdf(
                page_num, section_idx, y_start, y_end
            )
            section_pdfs.append(section_pdf_path)
            self.log(f"    Saved: {section_pdf_path.name}")
        
        self.log(f"\nSuccessfully created {len(section_pdfs)} section PDFs!")
        self.log(f"Output directory: {self.sections_dir.absolute()}")
        
        # Save log file
        log_path = self.output_dir / "step1_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        
        self.log(f"Log saved to: {log_path}")
        
        return section_pdfs
    
    def close(self):
        """Close PDF documents."""
        self.fitz_doc.close()
        self.plumber_doc.close()


def main():
    """Main entry point."""
    import sys
    
    # Use PDF from data/input directory
    pdf_path = "data/input/pid-legend.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        print("Please ensure pid-legend.pdf is in the data/input/ directory")
        sys.exit(1)
    
    splitter = SectionSplitter(pdf_path)
    
    try:
        section_pdfs = splitter.split_pdf_into_sections(0)  # Assuming single page PDF
        print(f"\nSuccessfully split PDF into {len(section_pdfs)} sections!")
    except Exception as e:
        print(f"Error during section splitting: {e}")
        import traceback
        traceback.print_exc()
    finally:
        splitter.close()


if __name__ == "__main__":
    main()

