"""
Step 3: Extract symbols from each column PDF with ground truth labels
Output: Symbol images with ground truth labels as filenames
Filename format: Replace spaces with underscores, preserve numbers
"""

import re
import json
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class SymbolExtractor:
    """Extracts symbols from column PDFs with ground truth labels as filenames."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # Setup directories
        self.data_dir = self.base_dir / "data"
        self.columns_dir = self.data_dir / "columns"
        self.symbols_dir = self.data_dir / "symbols"
        self.output_dir = self.base_dir / "output"
        
        # Create directories
        for dir_path in [self.columns_dir, self.symbols_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.extracted_symbols = []
        self.metadata = []
        self.log_entries = []
    
    def log(self, message: str):
        """Log a message to console and store for file output."""
        print(message)
        self.log_entries.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize label name for use as filename.
        Replace spaces with underscores, preserve all numbers and special characters.
        Only remove invalid filename characters.
        """
        if not name:
            return "unnamed_symbol"
        
        # Replace invalid filename characters (Windows/Unix invalid chars)
        invalid_chars = r'[<>:"/\\|?*]'
        name = re.sub(invalid_chars, '_', name)
        
        # Replace spaces with underscores (preserve numbers and other characters)
        name = re.sub(r'\s+', '_', name)
        
        # Remove leading/trailing dots and underscores
        name = name.strip('. _')
        
        return name if name else "unnamed_symbol"
    
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
    
    def extract_text_from_column(self, column_img: Image.Image, scale_x: float, scale_y: float) -> List[Dict]:
        """Extract text from column image using OCR, grouping words into phrases."""
        text_objects = []
        words = []
        
        try:
            # Use OCR with bounding boxes since column PDFs are image-only
            ocr_data = pytesseract.image_to_data(column_img, output_type=pytesseract.Output.DICT, config='--psm 6')
            
            # Convert OCR results to word format similar to pdfplumber
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                conf = ocr_data['conf'][i]
                
                # Filter low confidence and empty text
                try:
                    conf_int = int(conf) if conf != '-1' else 0
                except (ValueError, TypeError):
                    conf_int = 0
                
                if text and conf_int > 30:  # Confidence threshold
                    x0 = ocr_data['left'][i] / scale_x
                    y0 = ocr_data['top'][i] / scale_y
                    x1 = (ocr_data['left'][i] + ocr_data['width'][i]) / scale_x
                    y1 = (ocr_data['top'][i] + ocr_data['height'][i]) / scale_y
                    
                    words.append({
                        'text': text,
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1,
                        'top': y0,
                        'bottom': y1
                    })
        except Exception as e:
            self.log(f"    OCR error: {e}")
            return text_objects
        
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
                                    'bbox': (x0, y0, x1, y1)
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
                            'bbox': (x0, y0, x1, y1)
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
                    'bbox': (x0, y0, x1, y1)
                })
        
        return text_objects
    
    def find_symbol_bounds(self, column_img: Image.Image, text_x: float, text_y: float, 
                          text_height: float, scale_x: float, scale_y: float) -> Optional[Tuple[int, int, int, int]]:
        """Find the actual bounds of a symbol to the left of text label."""
        img_text_x = int(text_x * scale_x)
        img_text_y = int(text_y * scale_y)
        img_text_h = int(text_height * scale_y)
        
        # Search to the left of text (symbols are on the left, labels on the right)
        search_left = max(0, img_text_x - int(120 * scale_x))
        search_right = img_text_x - int(8 * scale_x)  # Leave gap from text
        
        text_center_y = img_text_y + img_text_h / 2
        search_vertical_range = max(int(img_text_h * 1.5), int(30 * scale_y))
        search_top = max(0, int(text_center_y - search_vertical_range / 2))
        search_bottom = min(column_img.height, int(text_center_y + search_vertical_range / 2))
        
        if search_right <= search_left or search_bottom <= search_top:
            return None
        
        search_region = column_img.crop((search_left, search_top, search_right, search_bottom))
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
    
    def process_column_pdf(self, column_pdf_path: Path) -> List[Dict]:
        """Process a single column PDF to extract symbols."""
        # Extract section and column indices from filename (section_XX_YY.pdf)
        try:
            parts = column_pdf_path.stem.split('_')
            section_idx = int(parts[1]) - 1  # Convert to 0-based
            column_idx = int(parts[2]) - 1
        except (ValueError, IndexError):
            section_idx = -1
            column_idx = -1
        
        # Open column PDF
        column_fitz_doc = fitz.open(str(column_pdf_path))
        
        try:
            page = column_fitz_doc[0]  # Column PDF has only one page
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Render column as image
            zoom = 3.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            column_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            scale_x = pix.width / page_width
            scale_y = pix.height / page_height
            
            # Extract text objects using OCR (column PDFs are image-only)
            text_objects = self.extract_text_from_column(column_img, scale_x, scale_y)
            
            # Debug: Log text extraction results
            if len(text_objects) == 0:
                # Only log for first few columns to avoid spam
                if column_idx < 3:
                    self.log(f"    No text objects found in {column_pdf_path.name}")
            else:
                if column_idx < 3:
                    self.log(f"    Found {len(text_objects)} text objects in {column_pdf_path.name}")
                    # Show first few labels
                    for i, txt in enumerate(text_objects[:3]):
                        self.log(f"      Text {i+1}: '{txt['text']}' at ({txt['bbox'][0]:.1f}, {txt['bbox'][1]:.1f})")
            
            # Filter out non-symbol labels
            skip_words = {'and', 'or', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'with',
                         'from', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
            
            skip_patterns = [
                r'^\d+$',  # Just numbers
                r'^\d+\s+\d+',  # Multiple numbers
                r'^[A-Z]{1,3}\s+\d+$',  # Codes like "PT 55"
            ]
            
            section_names_lower = [name.lower() for name in [
                "Instrument", "Valves", "Pumps", "Vessels", "Filters", "Compressors",
                "Heat Exchanges", "Dryers", "General", "Mixers", "Crushers", "Centrifuges",
                "Motors", "Peripheral", "Piping and Connecting Shapes"
            ]]
            
            column_symbols = []
            processed_regions = set()
            
            # Debug: Track filtering
            filtered_count = 0
            no_symbol_count = 0
            
            for text_obj in text_objects:
                text_bbox = text_obj['bbox']
                label = text_obj['text'].strip()
                
                # Skip if label is too short or matches skip patterns
                if len(label) < 2:
                    filtered_count += 1
                    continue
                
                if any(re.match(pattern, label) for pattern in skip_patterns):
                    filtered_count += 1
                    continue
                
                label_lower = label.lower()
                
                # Skip section headers
                if label_lower in section_names_lower:
                    filtered_count += 1
                    continue
                
                # Skip common non-symbol words
                if label_lower in skip_words:
                    filtered_count += 1
                    continue
                
                # Find symbol bounds to the left of text label
                symbol_bounds = self.find_symbol_bounds(
                    column_img,
                    text_bbox[0],  # text x position
                    text_bbox[1],  # text y position
                    text_bbox[3] - text_bbox[1],  # text height
                    scale_x,
                    scale_y
                )
                
                if symbol_bounds is None:
                    no_symbol_count += 1
                    continue
                
                img_x0, img_y0, img_x1, img_y1 = symbol_bounds
                
                region_key = (round(img_x0/10)*10, round(img_y0/10)*10, round(img_x1/10)*10, round(img_y1/10)*10)
                if region_key in processed_regions:
                    continue
                
                if img_x1 > img_x0 and img_y1 > img_y0:
                    try:
                        symbol_img = column_img.crop((img_x0, img_y0, img_x1, img_y1))
                        
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
                                    'section_idx': section_idx,
                                    'column_idx': column_idx,
                                    'column_file': column_pdf_path.name
                                })
                                processed_regions.add(region_key)
                    except Exception as e:
                        continue
            
            # Debug logging for first few columns
            if column_idx < 3:
                self.log(f"    Column {column_pdf_path.name}: {len(text_objects)} texts, {filtered_count} filtered, {no_symbol_count} no symbol found, {len(column_symbols)} extracted")
            
            return column_symbols
            
        finally:
            column_fitz_doc.close()
    
    def process_all_columns(self) -> List[Dict]:
        """Process all 75 column PDFs and extract symbols."""
        self.log("="*60)
        self.log("STEP 3: Symbol Extraction")
        self.log("="*60)
        
        # Find all column PDFs
        column_files = sorted(self.columns_dir.glob("section_*.pdf"))
        
        if not column_files:
            self.log("Error: No column PDFs found in data/columns/")
            self.log("Please run step2_split_columns.py first")
            return []
        
        self.log(f"Found {len(column_files)} column PDFs to process")
        
        all_symbols = []
        
        for col_idx, column_file in enumerate(column_files):
            if (col_idx + 1) % 10 == 0:
                self.log(f"  Processing column {col_idx + 1}/{len(column_files)}: {column_file.name}")
            
            column_symbols = self.process_column_pdf(column_file)
            all_symbols.extend(column_symbols)
            
            if column_symbols:
                self.log(f"    Extracted {len(column_symbols)} symbols from {column_file.name}")
        
        self.extracted_symbols = all_symbols
        return all_symbols
    
    def save_symbols(self):
        """Save extracted symbols with ground truth labels as filenames, organized by section."""
        self.log(f"\nSaving {len(self.extracted_symbols)} symbols...")
        
        # Group symbols by section index for organization
        symbols_by_section = {}
        for symbol in self.extracted_symbols:
            section_idx = symbol.get('section_idx', -1)
            if section_idx not in symbols_by_section:
                symbols_by_section[section_idx] = []
            symbols_by_section[section_idx].append(symbol)
        
        # Create section folders and save symbols
        for section_idx, symbols in symbols_by_section.items():
            # Create section-specific folder: section_01_symbols, section_02_symbols, etc.
            section_folder_name = f"section_{section_idx + 1:02d}_symbols"
            section_dir = self.symbols_dir / section_folder_name
            section_dir.mkdir(parents=True, exist_ok=True)
            
            # Track filename counts per section to handle duplicates
            filename_count = {}
            
            for symbol in symbols:
                label = symbol['label']
                sanitized_name = self.sanitize_filename(label)
                
                # Handle duplicates within the same section
                key = f"{section_idx}_{sanitized_name}"
                if key in filename_count:
                    filename_count[key] += 1
                    filename = f"{sanitized_name}_{filename_count[key]}.png"
                else:
                    filename_count[key] = 0
                    filename = f"{sanitized_name}.png"
                
                filepath = section_dir / filename
                symbol['image'].save(filepath, 'PNG')
                
                # Save relative path from symbols_dir for metadata
                relative_path = f"{section_folder_name}/{filename}"
                
                self.metadata.append({
                    'filename': filename,
                    'section_folder': section_folder_name,
                    'relative_path': relative_path,
                    'original_label': symbol['label'],
                    'section_idx': section_idx,
                    'column_idx': symbol.get('column_idx', -1),
                    'column_file': symbol.get('column_file', 'Unknown')
                })
                
                self.log(f"  Saved: {relative_path} (from label: {label})")
            
            self.log(f"  Section {section_idx + 1}: Saved {len(symbols)} symbols to {section_folder_name}/")
        
        # Save metadata file
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nMetadata saved to: {metadata_path}")
        
        # Save log file
        log_path = self.output_dir / "step3_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        
        self.log(f"Log saved to: {log_path}")


def main():
    """Main entry point."""
    extractor = SymbolExtractor()
    
    try:
        symbols = extractor.process_all_columns()
        extractor.save_symbols()
        
        print(f"\n{'='*60}")
        print(f"Extraction Complete!")
        print(f"Total symbols extracted: {len(symbols)}")
        print(f"Output directory: {extractor.symbols_dir.absolute()}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Error during symbol extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

