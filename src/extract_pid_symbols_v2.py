"""
P&ID Symbols Extraction Script V2
Two-phase extraction: First splits PDF into 15 sections, then processes each section separately.
"""

import os
import re
import json
from itertools import combinations
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class PIDSymbolExtractorV2:
    """Extracts P&ID symbols from PDF using two-phase approach:
    Phase 1: Split page into 15 sections and save as separate PDFs
    Phase 2: Process each section PDF to extract symbols from 5 columns
    """
    
    def __init__(self, pdf_path: str, base_dir: str = "."):
        self.pdf_path = Path(pdf_path)
        self.base_dir = Path(base_dir)
        
        # Setup directories
        self.data_dir = self.base_dir / "data"
        self.input_dir = self.data_dir / "input"
        self.sections_dir = self.data_dir / "sections"
        self.symbols_dir = self.data_dir / "symbols"
        self.output_dir = self.base_dir / "output"
        
        # Create directories
        for dir_path in [self.input_dir, self.sections_dir, self.symbols_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Open PDF with both libraries
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.plumber_doc = pdfplumber.open(str(self.pdf_path))
        
        self.extracted_symbols = []
        self.metadata = []
        self.log_entries = []
    
    def log(self, message: str):
        """Log a message to console and store for file output."""
        print(message)
        self.log_entries.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize label name for use as filename."""
        if not name:
            return "unnamed_symbol"
        
        # Replace invalid filename characters
        invalid_chars = r'[<>:"/\\|?*]'
        name = re.sub(invalid_chars, '_', name)
        
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        
        # Remove leading/trailing dots and underscores
        name = name.strip('. _')
        
        return name if name else "unnamed_symbol"
    
    def extract_text_from_page(self, page_num: int) -> List[Dict]:
        """Extract text from PDF page with positions, grouping words into phrases."""
        page = self.plumber_doc.pages[page_num]
        text_objects = []
        
        words = page.extract_words()
        if not words:
            return text_objects
        
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        current_phrase = []
        current_y = None
        y_tolerance = 5
        max_phrase_length = 4
        
        for word in sorted_words:
            word_text = word['text'].strip()
            word_y = (word['top'] + word['bottom']) / 2
            
            if len(word_text) <= 1 and word_text not in ['I', 'O']:
                continue
            
            if current_y is None or abs(word_y - current_y) <= y_tolerance:
                if current_phrase:
                    last_word = current_phrase[-1]
                    gap = word['x0'] - last_word['x1']
                    avg_char_width = (last_word['x1'] - last_word['x0']) / max(len(last_word['text']), 1)
                    
                    if gap < avg_char_width * 2 and len(current_phrase) < max_phrase_length:
                        current_phrase.append(word)
                        continue
                    else:
                        if current_phrase:
                            phrase_text = ' '.join([w['text'] for w in current_phrase])
                            if not phrase_text.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                                x0 = min(w['x0'] for w in current_phrase)
                                y0 = min(w['top'] for w in current_phrase)
                                x1 = max(w['x1'] for w in current_phrase)
                                y1 = max(w['bottom'] for w in current_phrase)
                                
                                text_objects.append({
                                    'text': phrase_text,
                                    'bbox': (x0, y0, x1, y1),
                                    'page': page_num
                                })
                        current_phrase = [word]
                        current_y = word_y
                else:
                    current_phrase = [word]
                    current_y = word_y
            else:
                if current_phrase:
                    phrase_text = ' '.join([w['text'] for w in current_phrase])
                    if not phrase_text.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                        x0 = min(w['x0'] for w in current_phrase)
                        y0 = min(w['top'] for w in current_phrase)
                        x1 = max(w['x1'] for w in current_phrase)
                        y1 = max(w['bottom'] for w in current_phrase)
                        
                        text_objects.append({
                            'text': phrase_text,
                            'bbox': (x0, y0, x1, y1),
                            'page': page_num
                        })
                
                current_phrase = [word]
                current_y = word_y
        
        if current_phrase:
            phrase_text = ' '.join([w['text'] for w in current_phrase])
            if not phrase_text.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                x0 = min(w['x0'] for w in current_phrase)
                y0 = min(w['top'] for w in current_phrase)
                x1 = max(w['x1'] for w in current_phrase)
                y1 = max(w['bottom'] for w in current_phrase)
                
                text_objects.append({
                    'text': phrase_text,
                    'bbox': (x0, y0, x1, y1),
                    'page': page_num
                })
        
        return text_objects
    
    def clean_image(self, image: Image.Image) -> Image.Image:
        """Clean and crop image to remove unnecessary whitespace."""
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        non_white_mask = gray < 250
        coords = np.column_stack(np.where(non_white_mask))
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            h, w = img_array.shape[:2]
            pad_x = max(8, int((x_max - x_min) * 0.1))
            pad_y = max(8, int((y_max - y_min) * 0.1))
            
            x = max(0, x_min - pad_x)
            y = max(0, y_min - pad_y)
            x_end = min(w, x_max + pad_x)
            y_end = min(h, y_max + pad_y)
            
            image = image.crop((x, y, x_end, y_end))
        
        return image
    
    # ========== PHASE 1: Section Detection and Splitting ==========
    
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
    
    def detect_section_headers(self, page_num: int, page_start_y: float) -> List[Tuple[str, float]]:
        """Detect section header text and return list of (section_name, y_position) tuples."""
        page = self.plumber_doc.pages[page_num]
        words = page.extract_words()
        
        section_names = [
            "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
            "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
            "Motors", "Peripheral", "Piping and Connecting Shapes"
        ]
        
        words_below_start = [w for w in words if w['top'] > page_start_y]
        
        phrases = []
        current_phrase = []
        current_y = None
        y_tolerance = 5
        
        for word in sorted(words_below_start, key=lambda w: (w['top'], w['x0'])):
            word_y = (word['top'] + word['bottom']) / 2
            
            if current_y is None or abs(word_y - current_y) <= y_tolerance:
                if current_phrase:
                    last_word = current_phrase[-1]
                    gap = word['x0'] - last_word['x1']
                    avg_char_width = (last_word['x1'] - last_word['x0']) / max(len(last_word['text']), 1)
                    
                    if gap < avg_char_width * 3:
                        current_phrase.append(word)
                        continue
                    else:
                        phrase_text = ' '.join([w['text'] for w in current_phrase])
                        phrase_y = min(w['top'] for w in current_phrase)
                        phrases.append((phrase_text, phrase_y))
                        current_phrase = [word]
                        current_y = word_y
                else:
                    current_phrase = [word]
                    current_y = word_y
            else:
                if current_phrase:
                    phrase_text = ' '.join([w['text'] for w in current_phrase])
                    phrase_y = min(w['top'] for w in current_phrase)
                    phrases.append((phrase_text, phrase_y))
                current_phrase = [word]
                current_y = word_y
        
        if current_phrase:
            phrase_text = ' '.join([w['text'] for w in current_phrase])
            phrase_y = min(w['top'] for w in current_phrase)
            phrases.append((phrase_text, phrase_y))
        
        detected_headers = []
        for phrase_text, phrase_y in phrases:
            phrase_lower = phrase_text.lower()
            for section_name in section_names:
                if section_name.lower() in phrase_lower or phrase_lower in section_name.lower():
                    detected_headers.append((section_name, phrase_y))
                    break
        
        detected_headers.sort(key=lambda x: x[1])
        return detected_headers
    
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
        
        headers = self.detect_section_headers(page_num, page_start_y)
        
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
            if len(headers) >= 1:
                boundaries.append((page_start_y, headers[0][1] - 10))
                for i in range(len(headers)):
                    y_start = headers[i][1] - 10
                    if i < len(headers) - 1:
                        y_end = headers[i + 1][1] - 10
                    else:
                        y_end = page_height * 0.98
                    boundaries.append((y_start, y_end))
            else:
                section_height = (page_height - page_start_y) / 15
                for i in range(15):
                    y_start = page_start_y + i * section_height
                    y_end = page_start_y + (i + 1) * section_height
                    boundaries.append((y_start, y_end))
        
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
        
        if headers and len(headers) >= len(boundaries):
            for i, (section_name, header_y) in enumerate(headers[:len(boundaries)]):
                if i < len(boundaries):
                    y_start, y_end = boundaries[i]
                    if abs(header_y - y_start) < 50:
                        boundaries[i] = (header_y - 5, y_end)
        
        return boundaries[:15]
    
    def save_section_as_pdf(self, page_num: int, section_idx: int, section_name: str, 
                           y_start: float, y_end: float) -> Path:
        """Save a section as a separate PDF file."""
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        section_height = y_end - y_start
        
        # Define the source rectangle (clip region) on the original page
        source_rect = fitz.Rect(0, y_start, page_width, y_end)
        
        # Render the section region as an image
        zoom = 2.0  # Use reasonable zoom for quality
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=source_rect)
        
        # Create a new PDF document
        section_doc = fitz.open()
        
        # Create a new page with the section dimensions
        section_page = section_doc.new_page(width=page_width, height=section_height)
        
        # Calculate the position to insert the image (centered or scaled to fit)
        # The pixmap dimensions in points
        img_width_pt = pix.width / zoom
        img_height_pt = pix.height / zoom
        
        # Insert the image into the page
        # Convert pixmap to image and insert it
        img_rect = fitz.Rect(0, 0, img_width_pt, img_height_pt)
        section_page.insert_image(img_rect, pixmap=pix)
        
        # Sanitize section name for filename
        sanitized_name = self.sanitize_filename(section_name)
        filename = f"section_{section_idx+1:02d}_{sanitized_name}.pdf"
        filepath = self.sections_dir / filename
        
        # Save the section PDF
        section_doc.save(str(filepath))
        section_doc.close()
        pix = None  # Free memory
        
        return filepath
    
    def split_page_into_sections(self, page_num: int) -> List[Tuple[Path, str, int]]:
        """Detect boundaries and save 15 section PDFs. Returns list of (pdf_path, section_name, section_idx)."""
        self.log(f"Phase 1: Splitting page {page_num + 1} into sections...")
        
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
        
        # Get section headers for naming
        headers = self.detect_section_headers(page_num, page_start_y)
        expected_section_names = [
            "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
            "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
            "Motors", "Peripheral", "Piping and Connecting Shapes"
        ]
        
        section_names = {}
        for i in range(15):
            if i < len(headers):
                section_names[i] = headers[i][0]
            elif i < len(expected_section_names):
                section_names[i] = expected_section_names[i]
            else:
                section_names[i] = f"Section_{i+1}"
        
        # Save each section as PDF
        section_pdfs = []
        for section_idx, (y_start, y_end) in enumerate(section_boundaries[:15]):
            section_name = section_names.get(section_idx, f"Section_{section_idx+1}")
            self.log(f"  Saving section {section_idx + 1}/15: {section_name} (Y: {y_start:.1f}-{y_end:.1f})")
            
            section_pdf_path = self.save_section_as_pdf(
                page_num, section_idx, section_name, y_start, y_end
            )
            section_pdfs.append((section_pdf_path, section_name, section_idx))
            self.log(f"    Saved: {section_pdf_path.name}")
        
        return section_pdfs
    
    # ========== PHASE 2: Section Processing ==========
    
    def detect_column_boundaries(self, section_img: Image.Image, section_bbox: Tuple) -> List[float]:
        """Detect 4 internal vertical boundaries that divide section into 5 columns."""
        gray = cv2.cvtColor(np.array(section_img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        section_height = section_img.height
        section_width = section_img.width
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        min_line_length = int(section_height * 0.6)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                               minLineLength=min_line_length,
                               maxLineGap=15)
        
        vertical_lines_x = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if x2 == x1:
                    angle = 90
                else:
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle > 85 or (x2 == x1):
                    x_avg = (x1 + x2) / 2
                    vertical_lines_x.append(x_avg)
        
        lines2 = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(section_height * 0.5))
        
        if lines2 is not None:
            for line in lines2:
                rho, theta = line[0]
                angle_rad = theta
                if angle_rad < np.pi/36 or angle_rad > 35*np.pi/36:
                    x_coord = rho
                    if 0 <= x_coord < section_width:
                        vertical_lines_x.append(x_coord)
        
        vertical_lines_x = sorted(set(vertical_lines_x))
        
        if len(vertical_lines_x) >= 4:
            target_positions = [section_width * 0.2, section_width * 0.4, 
                              section_width * 0.6, section_width * 0.8]
            
            selected = []
            used_indices = set()
            
            for target in target_positions:
                best_idx = None
                best_dist = float('inf')
                for idx, x in enumerate(vertical_lines_x):
                    if idx not in used_indices:
                        dist = abs(x - target)
                        if dist < best_dist and dist < section_width * 0.2:
                            best_dist = dist
                            best_idx = idx
                
                if best_idx is not None:
                    selected.append(vertical_lines_x[best_idx])
                    used_indices.add(best_idx)
            
            if len(selected) == 4:
                return sorted(selected)
        
        if len(vertical_lines_x) > 4:
            best_set = None
            best_score = float('inf')
            
            for combo in combinations(vertical_lines_x, 4):
                combo_sorted = sorted(combo)
                spacings = [combo_sorted[i+1] - combo_sorted[i] for i in range(3)]
                if len(spacings) == 3:
                    mean_spacing = sum(spacings) / 3
                    variance = sum((s - mean_spacing)**2 for s in spacings) / 3
                    if variance < best_score:
                        best_score = variance
                        best_set = combo_sorted
            
            if best_set:
                return best_set
        
        return [section_width * 0.2, section_width * 0.4, section_width * 0.6, section_width * 0.8]
    
    def find_symbol_bounds(self, page_img: Image.Image, text_x: int, text_y: int, 
                          text_height: int, scale_x: float, scale_y: float,
                          column_left: Optional[float] = None, 
                          column_right: Optional[float] = None) -> Optional[Tuple[int, int, int, int]]:
        """Find the actual bounds of a symbol to the left of text label."""
        img_text_x = int(text_x * scale_x)
        img_text_y = int(text_y * scale_y)
        img_text_h = int(text_height * scale_y)
        
        if column_left is not None:
            search_left = max(0, int(column_left * scale_x))
        else:
            search_left = max(0, img_text_x - int(120 * scale_x))
        
        search_right = img_text_x - int(8 * scale_x)
        
        if column_right is not None:
            search_right = min(search_right, int(column_right * scale_x))
        
        text_center_y = img_text_y + img_text_h / 2
        search_vertical_range = max(int(img_text_h * 1.5), int(30 * scale_y))
        search_top = max(0, int(text_center_y - search_vertical_range / 2))
        search_bottom = min(page_img.height, int(text_center_y + search_vertical_range / 2))
        
        if search_right <= search_left or search_bottom <= search_top:
            return None
        
        search_region = page_img.crop((search_left, search_top, search_right, search_bottom))
        search_array = np.array(search_region.convert('RGB'))
        gray = cv2.cvtColor(search_array, cv2.COLOR_RGB2GRAY)
        
        non_white = gray < 230
        
        if np.sum(non_white) < 100:
            return None
        
        coords = np.column_stack(np.where(non_white))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        binary = (gray < 230).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cleaned, connectivity=8)
        
        if num_labels < 2:
            return None
        
        largest_component_idx = 1
        largest_area = stats[1, 4]
        
        for i in range(2, num_labels):
            area = stats[i, 4]
            if area > largest_area:
                largest_area = area
                largest_component_idx = i
        
        if largest_area < 200:
            return None
        
        x_min_comp = stats[largest_component_idx, 0]
        y_min_comp = stats[largest_component_idx, 1]
        x_max_comp = x_min_comp + stats[largest_component_idx, 2]
        y_max_comp = y_min_comp + stats[largest_component_idx, 3]
        
        h, w = search_array.shape[:2]
        pad_x = max(3, int((x_max_comp - x_min_comp) * 0.05))
        pad_y = max(3, int((y_max_comp - y_min_comp) * 0.05))
        
        x_min = max(0, x_min_comp - pad_x)
        y_min = max(0, y_min_comp - pad_y)
        x_max = min(w, x_max_comp + pad_x)
        y_max = min(h, y_max_comp + pad_y)
        
        abs_x0 = search_left + x_min
        abs_y0 = search_top + y_min
        abs_x1 = search_left + x_max
        abs_y1 = search_top + y_max
        
        return (abs_x0, abs_y0, abs_x1, abs_y1)
    
    def extract_symbols_from_column(self, section_img: Image.Image, section_bbox: Tuple,
                                     text_objects: List[Dict], scale_x: float, scale_y: float,
                                     section_name: str, section_idx: int, col_x0: float, col_x1: float) -> List[Dict]:
        """Extract symbol-label pairs from a single column."""
        column_symbols = []
        processed_regions = set()
        
        col_y0, col_y1 = section_bbox[1], section_bbox[3]
        column_texts = []
        
        for text_obj in text_objects:
            text_x0, text_y0, text_x1, text_y1 = text_obj['bbox']
            text_center_x = (text_x0 + text_x1) / 2
            
            if col_x0 <= text_center_x <= col_x1:
                if col_y0 <= text_y0 <= col_y1:
                    column_texts.append(text_obj)
        
        skip_words = {'and', 'or', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'with',
                     'from', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        skip_patterns = [
            r'^\d+$',
            r'^\d+\s+\d+',
            r'^[A-Z]{1,3}\s+\d+$',
        ]
        
        section_names_lower = [name.lower() for name in [
            "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
            "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
            "Motors", "Peripheral", "Piping and Connecting Shapes"
        ]]
        
        for text_obj in column_texts:
            text_bbox = text_obj['bbox']
            label = text_obj['text'].strip()
            
            if len(label) < 2:
                continue
            
            if any(re.match(pattern, label) for pattern in skip_patterns):
                continue
            
            label_lower = label.lower()
            
            if label_lower in section_names_lower:
                continue
            
            if label_lower in skip_words:
                continue
            
            sect_x0, sect_y0, sect_x1, sect_y1 = section_bbox
            
            text_x_rel = text_bbox[0] - sect_x0
            text_y_rel = text_bbox[1] - sect_y0
            
            symbol_bounds = self.find_symbol_bounds(
                section_img,
                text_x_rel,
                text_y_rel,
                text_bbox[3] - text_bbox[1],
                scale_x,
                scale_y,
                column_left=col_x0 - sect_x0,
                column_right=col_x1 - sect_x0
            )
            
            if symbol_bounds is None:
                continue
            
            img_x0, img_y0, img_x1, img_y1 = symbol_bounds
            
            region_key = (round(img_x0/10)*10, round(img_y0/10)*10, round(img_x1/10)*10, round(img_y1/10)*10)
            if region_key in processed_regions:
                continue
            
            if img_x1 > img_x0 and img_y1 > img_y0:
                try:
                    symbol_img = section_img.crop((img_x0, img_y0, img_x1, img_y1))
                    
                    img_array = np.array(symbol_img.convert('RGB'))
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    binary = (gray < 230).astype(np.uint8) * 255
                    kernel = np.ones((3, 3), np.uint8)
                    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_cleaned, connectivity=8)
                    
                    if num_labels < 2:
                        continue
                    
                    largest_area = stats[1, 4]
                    for i in range(2, num_labels):
                        area = stats[i, 4]
                        if area > largest_area:
                            largest_area = area
                    
                    non_white_pixels = np.sum(gray < 230)
                    aspect_ratio = symbol_img.width / max(symbol_img.height, 1)
                    
                    if (largest_area > 150 and
                        non_white_pixels > 200 and
                        0.15 < aspect_ratio < 6.0 and
                        symbol_img.width > 20 and symbol_img.height > 20):
                        
                        symbol_img = self.clean_image(symbol_img)
                        
                        if symbol_img.width > 15 and symbol_img.height > 15:
                            column_symbols.append({
                                'image': symbol_img,
                                'label': label,
                                'section': section_name,
                                'section_idx': section_idx
                            })
                            processed_regions.add(region_key)
                except Exception as e:
                    continue
        
        return column_symbols
    
    def process_section_pdf(self, section_pdf_path: Path, section_name: str, section_idx: int) -> List[Dict]:
        """Process a single section PDF to extract symbols from 5 columns."""
        self.log(f"  Processing section {section_idx + 1}/15: {section_name}")
        
        # Open section PDF
        section_fitz_doc = fitz.open(str(section_pdf_path))
        section_plumber_doc = pdfplumber.open(str(section_pdf_path))
        
        try:
            page = section_fitz_doc[0]  # Section PDF has only one page
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Render section as image
            zoom = 3.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            section_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            scale_x = pix.width / page_width
            scale_y = pix.height / page_height
            
            # Extract text objects
            text_objects = []
            plumber_page = section_plumber_doc.pages[0]
            words = plumber_page.extract_words()
            
            if words:
                sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
                current_phrase = []
                current_y = None
                y_tolerance = 5
                max_phrase_length = 4
                
                for word in sorted_words:
                    word_text = word['text'].strip()
                    word_y = (word['top'] + word['bottom']) / 2
                    
                    if len(word_text) <= 1 and word_text not in ['I', 'O']:
                        continue
                    
                    if current_y is None or abs(word_y - current_y) <= y_tolerance:
                        if current_phrase:
                            last_word = current_phrase[-1]
                            gap = word['x0'] - last_word['x1']
                            avg_char_width = (last_word['x1'] - last_word['x0']) / max(len(last_word['text']), 1)
                            
                            if gap < avg_char_width * 2 and len(current_phrase) < max_phrase_length:
                                current_phrase.append(word)
                                continue
                            else:
                                if current_phrase:
                                    phrase_text = ' '.join([w['text'] for w in current_phrase])
                                    if not phrase_text.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                                        x0 = min(w['x0'] for w in current_phrase)
                                        y0 = min(w['top'] for w in current_phrase)
                                        x1 = max(w['x1'] for w in current_phrase)
                                        y1 = max(w['bottom'] for w in current_phrase)
                                        
                                        text_objects.append({
                                            'text': phrase_text,
                                            'bbox': (x0, y0, x1, y1),
                                            'page': 0
                                        })
                                current_phrase = [word]
                                current_y = word_y
                        else:
                            current_phrase = [word]
                            current_y = word_y
                    else:
                        if current_phrase:
                            phrase_text = ' '.join([w['text'] for w in current_phrase])
                            if not phrase_text.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                                x0 = min(w['x0'] for w in current_phrase)
                                y0 = min(w['top'] for w in current_phrase)
                                x1 = max(w['x1'] for w in current_phrase)
                                y1 = max(w['bottom'] for w in current_phrase)
                                
                                text_objects.append({
                                    'text': phrase_text,
                                    'bbox': (x0, y0, x1, y1),
                                    'page': 0
                                })
                        
                        current_phrase = [word]
                        current_y = word_y
                
                if current_phrase:
                    phrase_text = ' '.join([w['text'] for w in current_phrase])
                    if not phrase_text.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                        x0 = min(w['x0'] for w in current_phrase)
                        y0 = min(w['top'] for w in current_phrase)
                        x1 = max(w['x1'] for w in current_phrase)
                        y1 = max(w['bottom'] for w in current_phrase)
                        
                        text_objects.append({
                            'text': phrase_text,
                            'bbox': (x0, y0, x1, y1),
                            'page': 0
                        })
            
            # Detect column boundaries
            self.log(f"    Detecting column boundaries...")
            section_bbox = (0, 0, page_width, page_height)
            column_boundaries_img = self.detect_column_boundaries(section_img, section_bbox)
            
            col_boundaries_pdf = [b / scale_x for b in column_boundaries_img]
            col_boundaries_pdf = sorted([max(0, min(b, page_width)) for b in col_boundaries_pdf])
            
            columns = []
            col_left = 0
            for col_right in col_boundaries_pdf:
                if col_right > col_left:
                    columns.append((col_left, col_right))
                    col_left = col_right
            
            if col_left < page_width:
                columns.append((col_left, page_width))
            
            self.log(f"    Found {len(columns)} columns")
            
            # Extract symbols from each column
            section_symbols = []
            for col_idx, (col_x0, col_x1) in enumerate(columns):
                column_symbols = self.extract_symbols_from_column(
                    section_img,
                    section_bbox,
                    text_objects,
                    scale_x,
                    scale_y,
                    section_name,
                    section_idx,
                    col_x0,
                    col_x1
                )
                
                section_symbols.extend(column_symbols)
                
                if column_symbols:
                    self.log(f"      Column {col_idx + 1}: Extracted {len(column_symbols)} symbols")
            
            return section_symbols
            
        finally:
            section_fitz_doc.close()
            section_plumber_doc.close()
    
    def save_symbols(self):
        """Save extracted symbols with appropriate filenames."""
        self.log(f"\nSaving {len(self.extracted_symbols)} symbols...")
        
        filename_count = {}
        
        for idx, symbol in enumerate(self.extracted_symbols):
            label = symbol['label']
            sanitized_name = self.sanitize_filename(label)
            
            if sanitized_name in filename_count:
                filename_count[sanitized_name] += 1
                filename = f"{sanitized_name}_{filename_count[sanitized_name]}.png"
            else:
                filename_count[sanitized_name] = 0
                filename = f"{sanitized_name}.png"
            
            filepath = self.symbols_dir / filename
            symbol['image'].save(filepath, 'PNG')
            
            self.metadata.append({
                'filename': filename,
                'original_label': symbol['label'],
                'section': symbol.get('section', 'Unknown'),
                'section_idx': symbol.get('section_idx', -1)
            })
            
            self.log(f"  Saved: {filename} (from label: {label})")
        
        # Save metadata file
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nMetadata saved to: {metadata_path}")
        
        # Save log file
        log_path = self.output_dir / "extraction_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        
        self.log(f"Log saved to: {log_path}")
    
    def extract(self):
        """Main extraction method - two-phase workflow."""
        self.log(f"Extracting symbols from: {self.pdf_path}")
        self.log(f"Output directory: {self.output_dir}\n")
        
        # Phase 1: Split page into sections and save as PDFs
        self.log("="*60)
        self.log("PHASE 1: Section Splitting")
        self.log("="*60)
        
        section_pdfs = self.split_page_into_sections(0)  # Assuming single page PDF
        
        # Phase 2: Process each section PDF
        self.log("\n" + "="*60)
        self.log("PHASE 2: Section Processing")
        self.log("="*60)
        
        for section_pdf_path, section_name, section_idx in section_pdfs:
            section_symbols = self.process_section_pdf(section_pdf_path, section_name, section_idx)
            
            for symbol_data in section_symbols:
                self.extracted_symbols.append(symbol_data)
        
        # Save all symbols
        self.save_symbols()
        
        # Generate summary
        self.log(f"\n{'='*60}")
        self.log(f"Extraction Complete!")
        self.log(f"Total symbols extracted: {len(self.extracted_symbols)}")
        self.log(f"Output directory: {self.output_dir.absolute()}")
        self.log(f"{'='*60}")
        
        return len(self.extracted_symbols)
    
    def close(self):
        """Close PDF documents."""
        self.fitz_doc.close()
        self.plumber_doc.close()


def main():
    """Main entry point."""
    import sys
    
    # Use PDF from data/input directory
    pdf_path = "data/input/pid-legend.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        print("Please ensure pid-legend.pdf is in the data/input/ directory")
        sys.exit(1)
    
    extractor = PIDSymbolExtractorV2(pdf_path)
    
    try:
        count = extractor.extract()
        print(f"\nSuccessfully extracted {count} symbols!")
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()

