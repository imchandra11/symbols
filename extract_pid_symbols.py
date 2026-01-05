"""
P&ID Symbols Extraction Script
Extracts symbol images from PDF and saves them with ground truth labels.
"""

import os
import re
import json
import io
from itertools import combinations
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pytesseract


class PIDSymbolExtractor:
    """Extracts P&ID symbols from PDF and matches them with labels."""
    
    def __init__(self, pdf_path: str, output_dir: str = "symbols_dataset"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Open PDF with both libraries
        self.fitz_doc = fitz.open(pdf_path)
        self.plumber_doc = pdfplumber.open(pdf_path)
        
        self.extracted_symbols = []
        self.metadata = []
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize label name for use as filename, preserving all characters except invalid filename chars.
        
        Preserves all special characters and numbers from the ground truth label.
        Only replaces invalid filename characters and spaces (for file system compatibility).
        """
        if not name:
            return "unnamed_symbol"
        
        # Keep all characters as-is, only replace invalid filename characters
        # Windows/Unix invalid chars: < > : " / \ | ? *
        invalid_chars = r'[<>:"/\\|?*]'
        name = re.sub(invalid_chars, '_', name)
        
        # Replace spaces with underscores for file system compatibility
        # (spaces in filenames can cause issues in scripts/command line)
        name = re.sub(r'\s+', '_', name)
        
        # Replace leading/trailing dots and underscores (not allowed in filenames on Windows)
        name = name.strip('. _')
        
        return name if name else "unnamed_symbol"
    
    def extract_images_from_page(self, page_num: int) -> List[Dict]:
        """Extract all images from a PDF page with their positions."""
        page = self.fitz_doc[page_num]
        images = []
        
        image_list = page.get_images()
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = self.fitz_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Get image position on page
            image_rects = page.get_image_rects(xref)
            
            for rect in image_rects:
                images.append({
                    'image_bytes': image_bytes,
                    'ext': image_ext,
                    'bbox': (rect.x0, rect.y0, rect.x1, rect.y1),
                    'page': page_num,
                    'index': img_idx
                })
        
        return images
    
    def extract_text_from_page(self, page_num: int) -> List[Dict]:
        """Extract text from PDF page with positions, grouping words into phrases."""
        page = self.plumber_doc.pages[page_num]
        text_objects = []
        
        # Extract text with positions
        words = page.extract_words()
        
        # Group words into phrases (words on same line that are close together)
        if not words:
            return text_objects
        
        # Sort words by y-position (top to bottom), then by x-position (left to right)
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        current_phrase = []
        current_y = None
        y_tolerance = 5  # Words within 5 points vertically are on same line
        
        # Common P&ID label patterns (typically 1-3 words)
        max_phrase_length = 4  # Maximum words in a typical symbol label
        
        for word in sorted_words:
            word_text = word['text'].strip()
            word_y = (word['top'] + word['bottom']) / 2
            
            # Skip very short words that are likely punctuation or codes
            if len(word_text) <= 1 and word_text not in ['I', 'O']:
                continue
            
            if current_y is None or abs(word_y - current_y) <= y_tolerance:
                # Same line - check if word is close horizontally
                if current_phrase:
                    last_word = current_phrase[-1]
                    gap = word['x0'] - last_word['x1']
                    # If gap is small (less than 2x average character width), it's part of same phrase
                    avg_char_width = (last_word['x1'] - last_word['x0']) / max(len(last_word['text']), 1)
                    
                    # Also check if we've reached max phrase length
                    if gap < avg_char_width * 2 and len(current_phrase) < max_phrase_length:
                        current_phrase.append(word)
                        continue
                    else:
                        # Gap is too large or phrase too long - save current phrase and start new one
                        if current_phrase:
                            phrase_text = ' '.join([w['text'] for w in current_phrase])
                            # Filter out phrases that are just numbers or codes
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
                    # Start new phrase
                    current_phrase = [word]
                    current_y = word_y
            else:
                # New line - save previous phrase
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
                
                # Start new phrase
                current_phrase = [word]
                current_y = word_y
        
        # Don't forget the last phrase
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
    
    def calculate_distance(self, img_bbox: Tuple, text_bbox: Tuple) -> float:
        """Calculate distance between image and text bounding boxes."""
        img_center_x = (img_bbox[0] + img_bbox[2]) / 2
        img_center_y = (img_bbox[1] + img_bbox[3]) / 2
        
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
        # Euclidean distance
        distance = np.sqrt((img_center_x - text_center_x)**2 + (img_center_y - text_center_y)**2)
        return distance
    
    def find_label_for_image(self, img_bbox: Tuple, text_objects: List[Dict], 
                            page_width: float, page_height: float) -> Optional[str]:
        """Find the closest text label for an image. Labels are expected to be on the RIGHT side of symbols."""
        if not text_objects:
            return None
        
        img_center_x = (img_bbox[0] + img_bbox[2]) / 2
        img_center_y = (img_bbox[1] + img_bbox[3]) / 2
        img_right = img_bbox[2]  # Right edge of image
        
        min_distance = float('inf')
        best_label = None
        
        # Look for text near the image (within reasonable distance)
        # In legends, labels are typically to the right, so prioritize that
        search_radius = min(page_width, page_height) * 0.3
        
        # First pass: STRICTLY prefer labels to the RIGHT of the symbol
        for text_obj in text_objects:
            text_left = text_obj['bbox'][0]  # Left edge of text
            text_center_x = (text_obj['bbox'][0] + text_obj['bbox'][2]) / 2
            text_center_y = (text_obj['bbox'][1] + text_obj['bbox'][3]) / 2
            
            # Label must be to the right of the symbol (with small tolerance)
            if text_left < img_right - 10:  # Text starts before image ends - not to the right
                continue
            
            # Calculate distance
            distance = self.calculate_distance(img_bbox, text_obj['bbox'])
            
            # Also check vertical alignment (labels should be roughly aligned with symbols)
            vertical_overlap = not (text_obj['bbox'][3] < img_bbox[1] or text_obj['bbox'][1] > img_bbox[3])
            
            # Prioritize labels that are:
            # 1. To the right of the symbol
            # 2. Vertically aligned (within reasonable range)
            # 3. Close horizontally (not too far to the right)
            horizontal_gap = text_left - img_right
            max_horizontal_gap = min(page_width * 0.15, 100)  # Max 15% of page width or 100 points
            
            if (distance < search_radius and 
                vertical_overlap and 
                horizontal_gap < max_horizontal_gap and
                distance < min_distance):
                min_distance = distance
                best_label = text_obj['text']
        
        # If no label found strictly to the right, look for labels slightly to the right or below
        if best_label is None:
            for text_obj in text_objects:
                text_center_x = (text_obj['bbox'][0] + text_obj['bbox'][2]) / 2
                text_center_y = (text_obj['bbox'][1] + text_obj['bbox'][3]) / 2
                
                distance = self.calculate_distance(img_bbox, text_obj['bbox'])
                
                # Text should be to the right or below (but prefer right)
                if (text_center_x > img_center_x or text_center_y > img_center_y) and distance < min_distance:
                    min_distance = distance
                    best_label = text_obj['text']
        
        return best_label
    
    def extract_text_via_ocr(self, image: Image.Image, region: Tuple = None) -> str:
        """Extract text from image using OCR as fallback."""
        try:
            if region:
                image = image.crop(region)
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def clean_image(self, image: Image.Image) -> Image.Image:
        """Clean and crop image to remove unnecessary whitespace, preserving full symbols."""
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find bounding box of non-white content (threshold: not pure white)
        # Use a slightly higher threshold to preserve light gray lines
        non_white_mask = gray < 250
        coords = np.column_stack(np.where(non_white_mask))
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding (10% of dimensions, minimum 8px to preserve symbol edges)
            h, w = img_array.shape[:2]
            pad_x = max(8, int((x_max - x_min) * 0.1))
            pad_y = max(8, int((y_max - y_min) * 0.1))
            
            x = max(0, x_min - pad_x)
            y = max(0, y_min - pad_y)
            x_end = min(w, x_max + pad_x)
            y_end = min(h, y_max + pad_y)
            
            # Crop image
            image = image.crop((x, y, x_end, y_end))
        
        return image
    
    def detect_top_section_boundary(self, page_num: int) -> float:
        """Detect and return Y-coordinate where top section (heading/description) ends.
        
        Returns the Y-coordinate that marks the end of the heading/description section
        and the start of symbol sections.
        """
        page = self.plumber_doc.pages[page_num]
        page_height = page.height
        
        # Analyze top 15% of page for heading/description
        top_portion_height = page_height * 0.15
        words = page.extract_words()
        
        # Find words in the top portion
        top_words = [w for w in words if w['top'] < top_portion_height]
        
        if not top_words:
            # If no words in top portion, assume top section is small
            return min(page_height * 0.1, 100)
        
        # Sort by Y position
        top_words_sorted = sorted(top_words, key=lambda w: w['top'])
        
        # Find the largest font size (likely heading)
        font_sizes = [w.get('size', 10) for w in top_words_sorted if 'size' in w]
        if font_sizes:
            max_font_size = max(font_sizes)
            # Heading likely has larger font
            heading_words = [w for w in top_words_sorted if w.get('size', 10) >= max_font_size * 0.9]
            if heading_words:
                heading_bottom = max(w['bottom'] for w in heading_words)
            else:
                heading_bottom = max(w['bottom'] for w in top_words_sorted[:5])  # First few words
        else:
            heading_bottom = max(w['bottom'] for w in top_words_sorted[:5])
        
        # Look for gap after description text
        # Get all words below heading to find where description ends
        all_words_below_heading = [w for w in words if w['top'] > heading_bottom]
        if all_words_below_heading:
            all_words_below_heading.sort(key=lambda w: w['top'])
            
            # Analyze gaps - find where there's a large gap indicating end of description
            for i in range(len(all_words_below_heading) - 1):
                gap = all_words_below_heading[i + 1]['top'] - all_words_below_heading[i]['bottom']
                # Large gap (>30 points) might indicate end of description
                if gap > 30:
                    # Check if next word looks like a section header (capitalized, specific patterns)
                    next_word_text = all_words_below_heading[i + 1]['text'].strip()
                    section_keywords = ['Instrument', 'Valves', 'Pumps', 'Vessels', 'Filters', 
                                      'Compressors', 'Heat', 'Dryers', 'Mixers', 'Crushers',
                                      'Centrifuges', 'Motors', 'Peripheral', 'Piping']
                    if any(keyword.lower() in next_word_text.lower() for keyword in section_keywords):
                        return all_words_below_heading[i]['bottom']
            
            # If no clear gap found, use end of first 10% of page
            return min(heading_bottom + 100, page_height * 0.15)
        else:
            return min(heading_bottom + 50, page_height * 0.1)
    
    def detect_horizontal_dotted_lines(self, page_img: Image.Image, page_start_y: float, 
                                       scale_y: float, page_width: float) -> List[float]:
        """Detect horizontal dashed/dotted lines on the page.
        
        Returns list of Y-coordinates (in PDF coordinates) where horizontal lines are detected.
        Should find approximately 30 lines (2 per section for 15 sections).
        """
        # Convert PIL image to numpy array
        img_array = np.array(page_img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Convert page_start_y to image coordinates
        page_start_y_img = int(page_start_y * scale_y)
        
        # Crop to region below top section
        gray_cropped = gray[page_start_y_img:, :]
        
        # Detect edges
        edges = cv2.Canny(gray_cropped, 50, 150, apertureSize=3)
        
        # Use morphological operations to connect dashed/dotted lines
        # Create horizontal kernel to connect horizontal dashes/dots
        kernel_horizontal = np.ones((1, 15), np.uint8)
        dilated = cv2.dilate(edges, kernel_horizontal, iterations=1)
        dilated = cv2.erode(dilated, kernel_horizontal, iterations=1)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100,
                               minLineLength=int(page_width * scale_y * 0.5),  # At least 50% of page width
                               maxLineGap=20)
        
        horizontal_lines_y = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Check if line is nearly horizontal (within ±3 degrees)
                if angle < 3 or angle > 177:
                    # Calculate average Y position
                    y_avg = (y1 + y2) / 2
                    # Convert to PDF coordinates and add back the offset
                    y_pdf = (y_avg + page_start_y_img) / scale_y
                    horizontal_lines_y.append(y_pdf)
        
        # Also try detecting by grouping horizontal line segments
        # Find horizontal contours that span significant width
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is horizontal (wide and short)
            if w > page_width * scale_y * 0.5 and h < 5:
                y_center = y + h / 2
                y_pdf = (y_center + page_start_y_img) / scale_y
                
                # Avoid duplicates (within 5 points)
                if not any(abs(existing - y_pdf) < 5 for existing in horizontal_lines_y):
                    horizontal_lines_y.append(y_pdf)
        
        # Sort by Y coordinate
        horizontal_lines_y = sorted(set(horizontal_lines_y))
        
        # Group lines that are very close together (within 3 points)
        grouped_lines = []
        for line_y in horizontal_lines_y:
            if not grouped_lines or abs(line_y - grouped_lines[-1]) > 3:
                grouped_lines.append(line_y)
            else:
                # Average with nearby line
                grouped_lines[-1] = (grouped_lines[-1] + line_y) / 2
        
        return grouped_lines
    
    def detect_section_headers(self, page_num: int, page_start_y: float) -> List[Tuple[str, float]]:
        """Detect section header text and return list of (section_name, y_position) tuples."""
        page = self.plumber_doc.pages[page_num]
        words = page.extract_words()
        
        # Known section names (exact order: 15 sections)
        section_names = [
            "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
            "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
            "Motors", "Peripheral", "Piping and Connecting Shapes"
        ]
        
        # Filter words below page_start_y
        words_below_start = [w for w in words if w['top'] > page_start_y]
        
        # Group words into phrases
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
                    
                    if gap < avg_char_width * 3:  # Words close together
                        current_phrase.append(word)
                        continue
                    else:
                        # Save current phrase
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
        
        # Add last phrase
        if current_phrase:
            phrase_text = ' '.join([w['text'] for w in current_phrase])
            phrase_y = min(w['top'] for w in current_phrase)
            phrases.append((phrase_text, phrase_y))
        
        # Match phrases to section names
        detected_headers = []
        for phrase_text, phrase_y in phrases:
            phrase_lower = phrase_text.lower()
            for section_name in section_names:
                if section_name.lower() in phrase_lower or phrase_lower in section_name.lower():
                    detected_headers.append((section_name, phrase_y))
                    break
        
        # Sort by Y position
        detected_headers.sort(key=lambda x: x[1])
        return detected_headers
    
    def refine_section_boundaries_with_gaps(self, initial_boundaries: List[Tuple[float, float]], 
                                           page_img: Image.Image, scale_y: float) -> List[Tuple[float, float]]:
        """Refine section boundaries by analyzing content density gaps."""
        if not initial_boundaries:
            return initial_boundaries
        
        # Convert page image to grayscale for gap analysis
        gray = cv2.cvtColor(np.array(page_img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        
        # Create vertical projection (sum of non-white pixels per row)
        non_white = gray < 240
        vertical_projection = np.sum(non_white, axis=1)  # Sum along horizontal axis
        
        # Find gaps (rows with minimal content)
        # Normalize projection
        if vertical_projection.max() > 0:
            normalized_proj = vertical_projection / vertical_projection.max()
        else:
            normalized_proj = vertical_projection
        
        # Find local minima (gaps between sections)
        gaps = []
        threshold = 0.1  # 10% of max content density
        
        for i in range(1, len(normalized_proj) - 1):
            if (normalized_proj[i] < threshold and 
                normalized_proj[i] < normalized_proj[i-1] and 
                normalized_proj[i] < normalized_proj[i+1]):
                # Convert to PDF coordinates
                gaps.append(i / scale_y)
        
        # Adjust boundaries based on gaps
        refined = []
        for y_start, y_end in initial_boundaries:
            # Find nearest gap at the start
            best_start = y_start
            best_end = y_end
            
            for gap_y in gaps:
                if abs(gap_y - y_start) < 20:  # Within 20 points
                    best_start = gap_y
                if abs(gap_y - y_end) < 20:
                    best_end = gap_y
            
            refined.append((best_start, best_end))
        
        return refined
    
    def detect_section_boundaries(self, page_num: int, page_start_y: float) -> List[Tuple[float, float]]:
        """Detect 15 section boundaries starting from page_start_y using horizontal dashed/dotted lines."""
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Render page for line detection
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        scale_y = pix.height / page.rect.height
        scale_x = pix.width / page.rect.width
        
        # Detect horizontal dashed/dotted lines
        print(f"    Detecting horizontal dashed/dotted lines...")
        horizontal_lines = self.detect_horizontal_dotted_lines(page_img, page_start_y, scale_y, page_width)
        print(f"    Found {len(horizontal_lines)} horizontal lines (expected ~30)")
        
        # Detect section headers for validation and naming
        headers = self.detect_section_headers(page_num, page_start_y)
        
        # Pair lines into section boundaries
        # Each section has 2 lines: start (top) and end (bottom)
        # Lines should be in pairs: line 1-2 = section 1, line 3-4 = section 2, etc.
        boundaries = []
        
        if len(horizontal_lines) >= 2:
            # Pair consecutive lines to form section boundaries
            for i in range(0, len(horizontal_lines) - 1, 2):
                y_start = horizontal_lines[i]
                y_end = horizontal_lines[i + 1] if i + 1 < len(horizontal_lines) else page_height * 0.98
                boundaries.append((y_start, y_end))
            
            # If we have odd number of lines, handle last one
            if len(horizontal_lines) % 2 == 1:
                # Last section extends to end of page
                y_start = horizontal_lines[-2]
                y_end = page_height * 0.98
                if boundaries:
                    boundaries[-1] = (y_start, y_end)
        else:
            # Fallback: use headers or divide evenly
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
                # Last resort: divide evenly
                section_height = (page_height - page_start_y) / 15
                for i in range(15):
                    y_start = page_start_y + i * section_height
                    y_end = page_start_y + (i + 1) * section_height
                    boundaries.append((y_start, y_end))
        
        # Ensure we have exactly 15 sections
        if len(boundaries) > 15:
            boundaries = boundaries[:15]
        elif len(boundaries) < 15:
            # If we have fewer, try to split larger sections
            while len(boundaries) < 15 and boundaries:
                # Find largest section and split it
                largest_idx = max(range(len(boundaries)), 
                                 key=lambda i: boundaries[i][1] - boundaries[i][0])
                y_start, y_end = boundaries[largest_idx]
                y_mid = (y_start + y_end) / 2
                boundaries[largest_idx] = (y_start, y_mid)
                boundaries.insert(largest_idx + 1, (y_mid, y_end))
        
        # Validate with section headers if available
        if headers and len(headers) >= len(boundaries):
            # Adjust boundaries to align with headers
            for i, (section_name, header_y) in enumerate(headers[:len(boundaries)]):
                if i < len(boundaries):
                    # Adjust start of section to be near header
                    y_start, y_end = boundaries[i]
                    # If header is close to boundary start, use it
                    if abs(header_y - y_start) < 50:
                        boundaries[i] = (header_y - 5, y_end)
        
        return boundaries[:15]
    
    def detect_column_boundaries(self, section_img: Image.Image, section_bbox: Tuple) -> List[float]:
        """Detect 4 internal vertical boundaries (lines) that divide section into 5 columns.
        
        Returns list of 4 x-coordinates: [x1, x2, x3, x4] (in image coordinates, relative to section)
        Columns are: section_left to x1, x1 to x2, x2 to x3, x3 to x4, x4 to section_right
        """
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(section_img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        section_height = section_img.height
        section_width = section_img.width
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect vertical lines using HoughLinesP
        # Lines should span significant portion of section height (>60%)
        min_line_length = int(section_height * 0.6)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                               minLineLength=min_line_length,
                               maxLineGap=15)
        
        vertical_lines_x = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                if x2 == x1:
                    angle = 90  # Perfectly vertical
                else:
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Check if line is nearly vertical (within ±5 degrees of 90)
                if angle > 85 or (x2 == x1):
                    # Calculate average X position
                    x_avg = (x1 + x2) / 2
                    vertical_lines_x.append(x_avg)
        
        # Also try HoughLines for better detection of continuous lines
        lines2 = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(section_height * 0.5))
        
        if lines2 is not None:
            for line in lines2:
                rho, theta = line[0]
                # Check if line is nearly vertical (theta close to 0 or π)
                angle_rad = theta
                if angle_rad < np.pi/36 or angle_rad > 35*np.pi/36:  # Within ±5 degrees of vertical
                    # Calculate x coordinate from rho and theta
                    # For vertical lines: x = rho / cos(theta), but cos(theta) ≈ 0
                    # Use approximation for near-vertical lines
                    x_coord = rho
                    if 0 <= x_coord < section_width:
                        vertical_lines_x.append(x_coord)
        
        # Remove duplicates and sort
        vertical_lines_x = sorted(set(vertical_lines_x))
        
        # Filter to get exactly 4 lines that are evenly spaced
        if len(vertical_lines_x) >= 4:
            # Select 4 lines that are most evenly spaced
            # Target positions: approximately 20%, 40%, 60%, 80% of width
            target_positions = [section_width * 0.2, section_width * 0.4, 
                              section_width * 0.6, section_width * 0.8]
            
            selected = []
            used_indices = set()
            
            for target in target_positions:
                # Find closest unused line
                best_idx = None
                best_dist = float('inf')
                for idx, x in enumerate(vertical_lines_x):
                    if idx not in used_indices:
                        dist = abs(x - target)
                        if dist < best_dist and dist < section_width * 0.2:  # Within 20% of target
                            best_dist = dist
                            best_idx = idx
                
                if best_idx is not None:
                    selected.append(vertical_lines_x[best_idx])
                    used_indices.add(best_idx)
            
            if len(selected) == 4:
                return sorted(selected)
        
        # If we have lines but not exactly 4, select the 4 that are most evenly spaced
        if len(vertical_lines_x) > 4:
            # Try all combinations to find most evenly spaced set
            best_set = None
            best_score = float('inf')
            
            for combo in combinations(vertical_lines_x, 4):
                combo_sorted = sorted(combo)
                # Calculate spacing variance (lower is better)
                spacings = [combo_sorted[i+1] - combo_sorted[i] for i in range(3)]
                if len(spacings) == 3:
                    mean_spacing = sum(spacings) / 3
                    variance = sum((s - mean_spacing)**2 for s in spacings) / 3
                    if variance < best_score:
                        best_score = variance
                        best_set = combo_sorted
            
            if best_set:
                return best_set
        
        # Fallback: divide evenly if detection fails
        return [section_width * 0.2, section_width * 0.4, section_width * 0.6, section_width * 0.8]
    
    def find_symbol_bounds(self, page_img: Image.Image, text_x: int, text_y: int, 
                          text_height: int, scale_x: float, scale_y: float,
                          column_left: Optional[float] = None, 
                          column_right: Optional[float] = None) -> Optional[Tuple[int, int, int, int]]:
        """Find the actual bounds of a symbol to the left of text label.
        
        In P&ID legends, labels are positioned to the RIGHT of symbols.
        This method searches to the LEFT of the given text position to find the corresponding symbol.
        
        Args:
            column_left: Left boundary of column (PDF coordinates) - if provided, search won't go beyond this
            column_right: Right boundary of column (PDF coordinates) - if provided, search won't go beyond this
        """
        # Convert to image coordinates
        img_text_x = int(text_x * scale_x)
        img_text_y = int(text_y * scale_y)
        img_text_h = int(text_height * scale_y)
        
        # Determine search boundaries
        if column_left is not None:
            search_left = max(0, int(column_left * scale_x))
        else:
            search_left = max(0, img_text_x - int(120 * scale_x))  # Search up to 120 points left
        
        search_right = img_text_x - int(8 * scale_x)  # Leave gap from text (8 points)
        
        if column_right is not None:
            # Ensure search doesn't exceed column boundary
            search_right = min(search_right, int(column_right * scale_x))
        
        # Narrow vertical search to align with text center (symbols are typically centered with their labels)
        text_center_y = img_text_y + img_text_h / 2
        search_vertical_range = max(int(img_text_h * 1.5), int(30 * scale_y))  # 1.5x text height or 30pt
        search_top = max(0, int(text_center_y - search_vertical_range / 2))
        search_bottom = min(page_img.height, int(text_center_y + search_vertical_range / 2))
        
        if search_right <= search_left or search_bottom <= search_top:
            return None
        
        # Extract search region
        search_region = page_img.crop((search_left, search_top, search_right, search_bottom))
        search_array = np.array(search_region.convert('RGB'))
        gray = cv2.cvtColor(search_array, cv2.COLOR_RGB2GRAY)
        
        # Use more aggressive threshold to detect symbol lines (not just text)
        # Symbols typically have darker lines than surrounding text
        non_white = gray < 230  # Lower threshold for symbol lines
        
        if np.sum(non_white) < 100:  # Need sufficient content
            return None
        
        # Find bounding box of symbol content
        coords = np.column_stack(np.where(non_white))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Use morphological operations to detect continuous symbol regions
        # This helps separate symbols from scattered text artifacts
        binary = (gray < 230).astype(np.uint8) * 255
        
        # Apply morphological opening to remove small noise and text artifacts
        kernel = np.ones((3, 3), np.uint8)
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find connected components to identify symbol regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cleaned, connectivity=8)
        
        if num_labels < 2:  # Only background
            return None
        
        # Find the largest connected component (likely the symbol)
        # Skip label 0 (background)
        # stats format: [left, top, width, height, area] for each component
        largest_component_idx = 1
        largest_area = stats[1, 4]  # Area is at index 4
        
        for i in range(2, num_labels):
            area = stats[i, 4]  # Area is at index 4
            if area > largest_area:
                largest_area = area
                largest_component_idx = i
        
        # Get bounding box of largest component
        # stats: [left, top, width, height, area]
        x_min_comp = stats[largest_component_idx, 0]  # left
        y_min_comp = stats[largest_component_idx, 1]  # top
        x_max_comp = x_min_comp + stats[largest_component_idx, 2]  # left + width
        y_max_comp = y_min_comp + stats[largest_component_idx, 3]  # top + height
        
        # Use the component-based bounds, but ensure we have a reasonable size
        if largest_area < 200:  # Minimum symbol area
            return None
        
        # Add small padding (5% of dimensions, min 3px)
        h, w = search_array.shape[:2]
        pad_x = max(3, int((x_max_comp - x_min_comp) * 0.05))
        pad_y = max(3, int((y_max_comp - y_min_comp) * 0.05))
        
        x_min = max(0, x_min_comp - pad_x)
        y_min = max(0, y_min_comp - pad_y)
        x_max = min(w, x_max_comp + pad_x)
        y_max = min(h, y_max_comp + pad_y)
        
        # Convert back to full page coordinates
        abs_x0 = search_left + x_min
        abs_y0 = search_top + y_min
        abs_x1 = search_left + x_max
        abs_y1 = search_top + y_max
        
        return (abs_x0, abs_y0, abs_x1, abs_y1)
    
    def extract_symbols_from_column(self, column_img: Image.Image, column_bbox: Tuple, 
                                     section_img: Image.Image, section_bbox: Tuple,
                                     text_objects: List[Dict], scale_x: float, scale_y: float,
                                     section_name: str, section_idx: int) -> List[Dict]:
        """Extract symbol-label pairs from a single column.
        
        Args:
            column_img: PIL Image of the column region
            column_bbox: Column bounding box in PDF coordinates (x0, y0, x1, y1)
            section_img: PIL Image of the entire section
            section_bbox: Section bounding box in PDF coordinates (x0, y0, x1, y1)
            text_objects: List of text objects (from extract_text_from_page)
            scale_x, scale_y: Scale factors for coordinate conversion
            section_name: Name of the section (for metadata)
            section_idx: Index of the section (0-14)
        """
        column_symbols = []
        processed_regions = set()
        
        # Filter text objects that are within this column
        col_x0, col_y0, col_x1, col_y1 = column_bbox
        column_texts = []
        
        for text_obj in text_objects:
            text_x0, text_y0, text_x1, text_y1 = text_obj['bbox']
            text_center_x = (text_x0 + text_x1) / 2
            
            # Check if text is within column horizontally
            if col_x0 <= text_center_x <= col_x1:
                # Check if text is within section vertically
                if col_y0 <= text_y0 <= col_y1:
                    column_texts.append(text_obj)
        
        # Filter out section headers and non-symbol labels
        skip_words = {'and', 'or', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'with',
                     'from', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^\d+\s+\d+',  # Multiple numbers
            r'^[A-Z]{1,3}\s+\d+$',  # Codes like "PT 55"
        ]
        
        # Known section names to skip (all 15 sections)
        section_names_lower = [name.lower() for name in [
            "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
            "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
            "Motors", "Peripheral", "Piping and Connecting Shapes"
        ]]
        
        for text_obj in column_texts:
            text_bbox = text_obj['bbox']
            label = text_obj['text'].strip()
            
            # Skip if label is too short or matches skip patterns
            if len(label) < 2:
                continue
            
            if any(re.match(pattern, label) for pattern in skip_patterns):
                continue
            
            label_lower = label.lower()
            
            # Skip section headers
            if label_lower in section_names_lower:
                continue
            
            # Skip common non-symbol words
            if label_lower in skip_words:
                continue
            
            # Convert text coordinates to be relative to section
            # text_bbox is in full page PDF coordinates, section_bbox is (x0, y0, x1, y1) in PDF coordinates
            sect_x0, sect_y0, sect_x1, sect_y1 = section_bbox
            
            # Find symbol bounds to the left of text label
            # Convert text coordinates to section-relative coordinates
            text_x_rel = text_bbox[0] - sect_x0  # Relative to section left edge
            text_y_rel = text_bbox[1] - sect_y0  # Relative to section top edge
            
            symbol_bounds = self.find_symbol_bounds(
                section_img,
                text_x_rel,  # text x position relative to section
                text_y_rel,  # text y position relative to section
                text_bbox[3] - text_bbox[1],  # text height
                scale_x,
                scale_y,
                column_left=col_x0 - sect_x0,  # Constrain search to column (relative to section)
                column_right=col_x1 - sect_x0
            )
            
            if symbol_bounds is None:
                continue
            
            img_x0, img_y0, img_x1, img_y1 = symbol_bounds
            
            # Create region key to avoid duplicates
            region_key = (round(img_x0/10)*10, round(img_y0/10)*10, round(img_x1/10)*10, round(img_y1/10)*10)
            if region_key in processed_regions:
                continue
            
            # Extract symbol region from section image
            if img_x1 > img_x0 and img_y1 > img_y0:
                try:
                    symbol_img = section_img.crop((img_x0, img_y0, img_x1, img_y1))
                    
                    # Verify it's a real symbol
                    img_array = np.array(symbol_img.convert('RGB'))
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    # Use connected components to check coherence
                    binary = (gray < 230).astype(np.uint8) * 255
                    kernel = np.ones((3, 3), np.uint8)
                    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_cleaned, connectivity=8)
                    
                    if num_labels < 2:
                        continue
                    
                    # Find largest component
                    largest_area = stats[1, 4]
                    for i in range(2, num_labels):
                        area = stats[i, 4]
                        if area > largest_area:
                            largest_area = area
                    
                    non_white_pixels = np.sum(gray < 230)
                    aspect_ratio = symbol_img.width / max(symbol_img.height, 1)
                    
                    # Filter criteria
                    if (largest_area > 150 and
                        non_white_pixels > 200 and
                        0.15 < aspect_ratio < 6.0 and
                        symbol_img.width > 20 and symbol_img.height > 20):
                        
                        # Clean the image
                        symbol_img = self.clean_image(symbol_img)
                        
                        if symbol_img.width > 15 and symbol_img.height > 15:
                            # Convert symbol bounds from section-relative to full page PDF coordinates
                            sect_x0, sect_y0, sect_x1, sect_y1 = section_bbox
                            bbox_pdf = (
                                (img_x0 / scale_x) + sect_x0,  # Add section offset
                                (img_y0 / scale_y) + sect_y0,
                                (img_x1 / scale_x) + sect_x0,
                                (img_y1 / scale_y) + sect_y0
                            )
                            
                            column_symbols.append({
                                'image': symbol_img,
                                'label': label,
                                'bbox': bbox_pdf,
                                'section': section_name,
                                'section_idx': section_idx
                            })
                            processed_regions.add(region_key)
                except Exception as e:
                    continue
        
        return column_symbols
    
    def extract_vector_symbols_from_page(self, page_num: int, text_objects: List[Dict]) -> List[Dict]:
        """Extract vector-drawn symbols by rendering page and identifying symbol regions.
        
        In P&ID legends, ground truth labels are positioned to the RIGHT of symbols.
        This method iterates through text labels and finds their corresponding symbols to the LEFT.
        """
        page = self.fitz_doc[page_num]
        
        # Render page at high resolution (3x for better quality)
        zoom = 3.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Scale factor for coordinate conversion
        scale_x = pix.width / page.rect.width
        scale_y = pix.height / page.rect.height
        
        vector_symbols = []
        processed_regions = set()  # Avoid duplicates
        
        # Filter text objects - skip common non-symbol words and phrases
        skip_words = {'and', 'or', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 
                     'from', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'PID', 'P&ID', 'Symbols', 'Legend', 'Industry', 'Standardized',
                     'Diagram', 'Standard', 'Detailed', 'Documentation', 'provides',
                     'standard', 'set', 'shapes', 'symbols', 'documenting', 'PFD',
                     'instrument', 'valves', 'pump', 'heating', 'exchanges', 'mixers',
                     'crushers', 'vessels', 'compressors', 'filters', 'motors',
                     'connecting', 'shapes', 'Major', 'Pipeline', 'Top', 'to',
                     'Requirements', 'Line', 'In', 'ri', 'lly', 'www.edrawsoft.com',
                     'Various', 'Mixers', 'Various Mixers'}
        
        # Skip patterns that indicate these are not symbol labels
        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^\d+\s+\d+',  # Multiple numbers
            r'^[A-Z]{1,3}\s+\d+',  # Codes like "PT 55"
            r'^\d+\s+[A-Z]',  # Numbers followed by letters
        ]
        
        for text_obj in text_objects:
            text_bbox = text_obj['bbox']
            label = text_obj['text'].strip()
            
            # Skip if label is too short
            if len(label) < 2:
                continue
            
            # Skip if matches skip patterns
            if any(re.match(pattern, label) for pattern in skip_patterns):
                continue
            
            # Skip if label is a skip word or starts/ends with parentheses
            label_lower = label.lower()
            label_words = label_lower.split()
            if (label_lower in skip_words or
                any(word in skip_words for word in label_words if len(label_words) > 3) or  # Skip if contains skip words and is too long
                label.replace('.', '').replace('-', '').replace(' ', '').isdigit() or
                label.startswith('(') or label.endswith(')') or
                label.startswith('www.') or 'edrawsoft' in label_lower):
                continue
            
            # Skip labels that are too long (likely multiple symbols combined)
            # But allow common multi-word symbols (up to 3 words is typical)
            if len(label_words) > 3:
                # Exception: allow certain common patterns
                common_patterns = ['temp indicator', 'flow transmitter', 'pressure gauge', 
                                  'gate valve', 'globe valve', 'angle valve', 'rotary valve',
                                  'needle valve', 'spray cooler', 'heat exchanger']
                if not any(pattern in label_lower for pattern in common_patterns):
                    continue
            
            # Find symbol bounds to the left of text label
            # In legends, labels are on the right, so we search left of the label
            symbol_bounds = self.find_symbol_bounds(
                page_img, 
                text_bbox[0],  # text x position (left edge of label)
                text_bbox[1],  # text y position (top of label)
                text_bbox[3] - text_bbox[1],  # text height
                scale_x, 
                scale_y
            )
            
            if symbol_bounds is None:
                continue
            
            img_x0, img_y0, img_x1, img_y1 = symbol_bounds
            
            # Create region key to avoid duplicates (rounded to avoid minor differences)
            region_key = (round(img_x0/10)*10, round(img_y0/10)*10, round(img_x1/10)*10, round(img_y1/10)*10)
            if region_key in processed_regions:
                continue
            
            # Extract symbol region
            if img_x1 > img_x0 and img_y1 > img_y0:
                try:
                    symbol_img = page_img.crop((img_x0, img_y0, img_x1, img_y1))
                    
                    # Verify it's a real symbol (has sufficient non-white content)
                    img_array = np.array(symbol_img.convert('RGB'))
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    # Use connected components to check if it's a coherent symbol
                    binary = (gray < 230).astype(np.uint8) * 255
                    kernel = np.ones((3, 3), np.uint8)
                    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_cleaned, connectivity=8)
                    
                    if num_labels < 2:  # Only background, no symbol
                        continue
                    
                    # Find largest component
                    # stats format: [left, top, width, height, area]
                    largest_area = stats[1, 4]  # Area is at index 4
                    for i in range(2, num_labels):
                        area = stats[i, 4]  # Area is at index 4
                        if area > largest_area:
                            largest_area = area
                    
                    non_white_pixels = np.sum(gray < 230)
                    
                    # Check aspect ratio - symbols are usually roughly square or slightly rectangular
                    aspect_ratio = symbol_img.width / max(symbol_img.height, 1)
                    
                    # More strict filtering: symbols should have significant coherent content
                    if (largest_area > 150 and  # Minimum coherent symbol area
                        non_white_pixels > 200 and  # Minimum total content
                        0.15 < aspect_ratio < 6.0 and  # Reasonable aspect ratio
                        symbol_img.width > 20 and symbol_img.height > 20):  # Minimum size
                        
                        # Clean the image (remove excess whitespace)
                        symbol_img = self.clean_image(symbol_img)
                        
                        # Final size check after cleaning
                        if symbol_img.width > 15 and symbol_img.height > 15:
                            vector_symbols.append({
                                'image': symbol_img,
                                'label': label,
                                'bbox': (img_x0/scale_x, img_y0/scale_y, img_x1/scale_x, img_y1/scale_y),
                                'page': page_num
                            })
                            processed_regions.add(region_key)
                except Exception as e:
                    continue
        
        return vector_symbols
    
    def process_page(self, page_num: int):
        """Process a single PDF page to extract symbols and labels using section-based approach."""
        print(f"Processing page {page_num + 1}/{len(self.fitz_doc)}...")
        
        # Get page dimensions
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Step 1: Detect and get Y-boundary of top section (heading/description)
        print("  Detecting top section boundary...")
        page_start_y = self.detect_top_section_boundary(page_num)
        print(f"  Top section ends at Y={page_start_y:.1f}")
        
        # Step 2: Detect 15 symbol sections (starting after top section)
        print("  Detecting section boundaries...")
        section_boundaries = self.detect_section_boundaries(page_num, page_start_y)
        print(f"  Found {len(section_boundaries)} sections")
        
        # Extract all text objects once
        text_objects = self.extract_text_from_page(page_num)
        
        # Get section headers for naming (exact order of 15 sections)
        headers = self.detect_section_headers(page_num, page_start_y)
        expected_section_names = [
            "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
            "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
            "Motors", "Peripheral", "Piping and Connecting Shapes"
        ]
        
        # Map section index to name
        section_names = {}
        for i in range(15):
            if i < len(headers):
                section_names[i] = headers[i][0]
            elif i < len(expected_section_names):
                section_names[i] = expected_section_names[i]
            else:
                section_names[i] = f"Section_{i+1}"
        
        # Step 3: Process each section
        for section_idx, (y_start, y_end) in enumerate(section_boundaries[:15]):
            section_name = section_names.get(section_idx, f"Section_{section_idx+1}")
            print(f"  Processing section {section_idx + 1}/15: {section_name} (Y: {y_start:.1f}-{y_end:.1f})")
            
            # Render section as image
            section_bbox_pdf = (0, y_start, page_width, y_end)
            zoom = 3.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render the section region
            clip_rect = fitz.Rect(0, y_start, page_width, y_end)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            section_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Scale factors
            scale_x = pix.width / page_width
            scale_y = pix.height / (y_end - y_start)
            
            # Step 3b: Detect 4 internal column boundaries (creates 5 columns)
            print(f"    Detecting column boundaries...")
            column_boundaries_img = self.detect_column_boundaries(section_img, section_bbox_pdf)
            
            # Convert column boundaries from image coordinates (pixels) to PDF coordinates (points)
            # Section image covers full page width, so boundaries are relative to section start (x=0)
            section_width = page_width
            col_boundaries_pdf = [b / scale_x for b in column_boundaries_img]
            
            # Ensure boundaries are within section and sorted
            col_boundaries_pdf = sorted([max(0, min(b, section_width)) for b in col_boundaries_pdf])
            
            # Create 5 columns: [section_left to x1, x1 to x2, x2 to x3, x3 to x4, x4 to section_right]
            columns = []
            col_left = 0
            for col_right in col_boundaries_pdf:
                if col_right > col_left:  # Valid column
                    columns.append((col_left, col_right))
                    col_left = col_right
            
            # Add last column if needed
            if col_left < section_width:
                columns.append((col_left, section_width))
            
            print(f"    Found {len(columns)} columns")
            
            # Step 3c: Extract symbols from each column
            for col_idx, (col_x0, col_x1) in enumerate(columns):
                column_bbox_pdf = (col_x0, y_start, col_x1, y_end)
                
                # Extract symbols from this column
                column_symbols = self.extract_symbols_from_column(
                    column_img=None,  # We'll use section_img with column coordinates
                    column_bbox=column_bbox_pdf,
                    section_img=section_img,
                    section_bbox=section_bbox_pdf,
                    text_objects=text_objects,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    section_name=section_name,
                    section_idx=section_idx
                )
                
                # Add to extracted symbols
                for symbol_data in column_symbols:
                    self.extracted_symbols.append({
                        'image': symbol_data['image'],
                        'label': symbol_data['label'],
                        'original_label': symbol_data['label'],
                        'page': page_num,
                        'bbox': symbol_data['bbox'],
                        'section': symbol_data.get('section', section_name),
                        'section_idx': symbol_data.get('section_idx', section_idx)
                    })
                
                if column_symbols:
                    print(f"      Column {col_idx + 1}: Extracted {len(column_symbols)} symbols")
    
    def save_symbols(self):
        """Save extracted symbols with appropriate filenames."""
        print(f"\nSaving {len(self.extracted_symbols)} symbols...")
        
        # Track filenames to handle duplicates
        filename_count = {}
        
        for idx, symbol in enumerate(self.extracted_symbols):
            label = symbol['label']
            sanitized_name = self.sanitize_filename(label)
            
            # Handle duplicates - since each ground truth is unique, use full label as key
            # This ensures we track duplicates based on exact label match
            if sanitized_name in filename_count:
                filename_count[sanitized_name] += 1
                filename = f"{sanitized_name}_{filename_count[sanitized_name]}.png"
            else:
                filename_count[sanitized_name] = 0
                filename = f"{sanitized_name}.png"
            
            # Save image
            filepath = self.output_dir / filename
            symbol['image'].save(filepath, 'PNG')
            
            # Store metadata
            self.metadata.append({
                'filename': filename,
                'original_label': symbol['original_label'],
                'page': symbol['page'] + 1,
                'bbox': symbol['bbox']
            })
            
            print(f"  Saved: {filename} (from label: {label})")
        
        # Save metadata file
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\nMetadata saved to: {metadata_path}")
    
    def extract(self):
        """Main extraction method."""
        print(f"Extracting symbols from: {self.pdf_path}")
        print(f"Output directory: {self.output_dir}\n")
        
        # Process all pages
        for page_num in range(len(self.fitz_doc)):
            self.process_page(page_num)
        
        # Save all symbols
        self.save_symbols()
        
        # Generate summary
        print(f"\n{'='*60}")
        print(f"Extraction Complete!")
        print(f"Total symbols extracted: {len(self.extracted_symbols)}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}")
        
        return len(self.extracted_symbols)
    
    def close(self):
        """Close PDF documents."""
        self.fitz_doc.close()
        self.plumber_doc.close()


def main():
    """Main entry point."""
    import sys
    
    pdf_path = "pid-legend.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    extractor = PIDSymbolExtractor(pdf_path)
    
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

