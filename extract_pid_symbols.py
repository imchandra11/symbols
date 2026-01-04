"""
P&ID Symbols Extraction Script
Extracts symbol images from PDF and saves them with ground truth labels.
"""

import os
import re
import json
import io
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
    
    def find_symbol_bounds(self, page_img: Image.Image, text_x: int, text_y: int, 
                          text_height: int, scale_x: float, scale_y: float) -> Optional[Tuple[int, int, int, int]]:
        """Find the actual bounds of a symbol to the left of text label.
        
        In P&ID legends, labels are positioned to the RIGHT of symbols.
        This method searches to the LEFT of the given text position to find the corresponding symbol.
        """
        # Convert to image coordinates
        img_text_x = int(text_x * scale_x)
        img_text_y = int(text_y * scale_y)
        img_text_h = int(text_height * scale_y)
        
        # Search region to the left of text - narrower vertical range to focus on symbol
        search_left = max(0, img_text_x - int(120 * scale_x))  # Search up to 120 points left
        search_right = img_text_x - int(8 * scale_x)  # Leave gap from text (8 points)
        
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
        """Process a single PDF page to extract symbols and labels."""
        print(f"Processing page {page_num + 1}/{len(self.fitz_doc)}...")
        
        # Get page dimensions
        page = self.fitz_doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Extract images and text
        images = self.extract_images_from_page(page_num)
        text_objects = self.extract_text_from_page(page_num)
        
        print(f"  Found {len(images)} embedded images and {len(text_objects)} text objects")
        
        # Match embedded images with labels
        for img_data in images:
            # Convert image bytes to PIL Image
            try:
                img = Image.open(io.BytesIO(img_data['image_bytes']))
                
                # Find label for this image
                label = self.find_label_for_image(
                    img_data['bbox'], 
                    text_objects, 
                    page_width, 
                    page_height
                )
                
                # If no label found, try OCR on the area around the image
                if not label:
                    # Extract text from region around image using OCR
                    page_image = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    pil_page = Image.frombytes("RGB", [page_image.width, page_image.height], 
                                              page_image.samples)
                    
                    # Crop region around image (expand bbox slightly)
                    x0, y0, x1, y1 = img_data['bbox']
                    expand = 50
                    region = (
                        max(0, int(x0 * 2 - expand)),
                        max(0, int(y0 * 2 - expand)),
                        min(pil_page.width, int(x1 * 2 + expand)),
                        min(pil_page.height, int(y1 * 2 + expand))
                    )
                    ocr_text = self.extract_text_via_ocr(pil_page, region)
                    if ocr_text:
                        # Try to extract meaningful label from OCR text
                        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
                        if lines:
                            label = lines[0]  # Use first non-empty line
                
                # Clean and process image
                img = self.clean_image(img)
                
                # Store symbol data
                self.extracted_symbols.append({
                    'image': img,
                    'label': label or f"Symbol_Page{page_num + 1}_Img{img_data['index']}",
                    'original_label': label,
                    'page': page_num,
                    'bbox': img_data['bbox']
                })
                
            except Exception as e:
                print(f"  Error processing image {img_data['index']}: {e}")
                continue
        
        # Extract vector-drawn symbols
        print(f"  Extracting vector symbols from text labels...")
        vector_symbols = self.extract_vector_symbols_from_page(page_num, text_objects)
        print(f"  Found {len(vector_symbols)} vector symbols")
        
        # Add vector symbols to extracted symbols
        for symbol_data in vector_symbols:
            self.extracted_symbols.append({
                'image': symbol_data['image'],
                'label': symbol_data['label'],
                'original_label': symbol_data['label'],
                'page': symbol_data['page'],
                'bbox': symbol_data['bbox']
            })
    
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

