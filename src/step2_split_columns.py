"""
Step 2: Split each section PDF into 5 vertical columns
Output: section_01_01.pdf, section_01_02.pdf, ..., section_15_05.pdf
"""

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import combinations
from datetime import datetime


class ColumnSplitter:
    """Splits each section PDF into 5 vertical columns."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # Setup directories
        self.data_dir = self.base_dir / "data"
        self.sections_dir = self.data_dir / "sections"
        self.columns_dir = self.data_dir / "columns"
        self.output_dir = self.base_dir / "output"
        
        # Create directories
        for dir_path in [self.sections_dir, self.columns_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.log_entries = []
    
    def log(self, message: str):
        """Log a message to console and store for file output."""
        print(message)
        self.log_entries.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def detect_column_boundaries(self, section_img: Image.Image, section_width: float) -> List[float]:
        """Detect 4 internal vertical boundaries that divide section into 5 columns."""
        gray = cv2.cvtColor(np.array(section_img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        section_height = section_img.height
        section_width_px = section_img.width
        
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
                    if 0 <= x_coord < section_width_px:
                        vertical_lines_x.append(x_coord)
        
        vertical_lines_x = sorted(set(vertical_lines_x))
        
        if len(vertical_lines_x) >= 4:
            target_positions = [section_width_px * 0.2, section_width_px * 0.4, 
                              section_width_px * 0.6, section_width_px * 0.8]
            
            selected = []
            used_indices = set()
            
            for target in target_positions:
                best_idx = None
                best_dist = float('inf')
                for idx, x in enumerate(vertical_lines_x):
                    if idx not in used_indices:
                        dist = abs(x - target)
                        if dist < best_dist and dist < section_width_px * 0.2:
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
        
        # Fallback: divide evenly
        return [section_width_px * 0.2, section_width_px * 0.4, section_width_px * 0.6, section_width_px * 0.8]
    
    def save_column_as_pdf(self, section_doc: fitz.Document, section_idx: int, column_idx: int,
                          col_x0: float, col_x1: float, section_height: float) -> Optional[Path]:
        """Save a column as a separate PDF file with naming: section_XX_YY.pdf"""
        try:
            page = section_doc[0]  # Section PDF has only one page
            page_width = page.rect.width
            
            # Validate column boundaries
            col_x0 = max(0, min(col_x0, page_width))
            col_x1 = max(col_x0 + 1, min(col_x1, page_width))  # Ensure at least 1 point width
            
            if col_x1 <= col_x0:
                self.log(f"      Warning: Invalid column boundaries for column {column_idx + 1}, skipping")
                return None
            
            # Define the source rectangle (clip region) for the column
            source_rect = fitz.Rect(col_x0, 0, col_x1, section_height)
            
            # Render the column region as an image
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = None
            try:
                pix = page.get_pixmap(matrix=mat, clip=source_rect)
            except Exception as e:
                self.log(f"      Warning: Failed to render column {column_idx + 1}: {e}")
                return None
            
            if pix is None or pix.width == 0 or pix.height == 0:
                self.log(f"      Warning: Empty pixmap for column {column_idx + 1}, skipping")
                if pix:
                    pix = None
                return None
            
            # Create a new PDF document
            column_doc = fitz.open()
            
            try:
                # Create a new page with the column dimensions
                column_width = col_x1 - col_x0
                if column_width <= 0 or section_height <= 0:
                    self.log(f"      Warning: Invalid dimensions for column {column_idx + 1}, skipping")
                    column_doc.close()
                    pix = None
                    return None
                
                column_page = column_doc.new_page(width=column_width, height=section_height)
                
                # Calculate the position to insert the image
                img_width_pt = pix.width / zoom
                img_height_pt = pix.height / zoom
                
                # Insert the image into the page
                img_rect = fitz.Rect(0, 0, img_width_pt, img_height_pt)
                try:
                    column_page.insert_image(img_rect, pixmap=pix)
                except Exception as e:
                    self.log(f"      Warning: Failed to insert image for column {column_idx + 1}: {e}")
                    column_doc.close()
                    pix = None
                    return None
                
                # Filename: section_01_01.pdf, section_01_02.pdf, etc.
                filename = f"section_{section_idx+1:02d}_{column_idx+1:02d}.pdf"
                filepath = self.columns_dir / filename
                
                # Save the column PDF
                column_doc.save(str(filepath))
                return filepath
                
            finally:
                column_doc.close()
                if pix:
                    pix = None  # Free memory
                    
        except Exception as e:
            self.log(f"      Error saving column {column_idx + 1}: {e}")
            return None
    
    def process_section_pdf(self, section_pdf_path: Path, section_idx: int) -> List[Path]:
        """Process a single section PDF to extract 5 columns."""
        self.log(f"  Processing section {section_idx + 1}/15: {section_pdf_path.name}")
        
        # Open section PDF
        section_doc = fitz.open(str(section_pdf_path))
        
        try:
            page = section_doc[0]  # Section PDF has only one page
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Render section as image for column detection
            zoom = 3.0
            mat = fitz.Matrix(zoom, zoom)
            pix = None
            try:
                pix = page.get_pixmap(matrix=mat)
                section_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Detect column boundaries
                self.log(f"    Detecting column boundaries...")
                column_boundaries_img = self.detect_column_boundaries(section_img, page_width)
                
                # Convert from image coordinates to PDF coordinates
                scale_x = pix.width / page_width
                col_boundaries_pdf = [b / scale_x for b in column_boundaries_img]
                col_boundaries_pdf = sorted([max(0, min(b, page_width)) for b in col_boundaries_pdf])
                
                # Create 5 columns
                columns = []
                col_left = 0
                for col_right in col_boundaries_pdf:
                    if col_right > col_left:
                        columns.append((col_left, col_right))
                        col_left = col_right
                
                if col_left < page_width:
                    columns.append((col_left, page_width))
                
                # Ensure we have exactly 5 columns
                if len(columns) > 5:
                    columns = columns[:5]
                elif len(columns) < 5:
                    # If we have fewer, divide evenly
                    while len(columns) < 5:
                        largest_idx = max(range(len(columns)), 
                                         key=lambda i: columns[i][1] - columns[i][0])
                        col_x0, col_x1 = columns[largest_idx]
                        col_mid = (col_x0 + col_x1) / 2
                        columns[largest_idx] = (col_x0, col_mid)
                        columns.insert(largest_idx + 1, (col_mid, col_x1))
                
                self.log(f"    Found {len(columns)} columns")
                
                # Save each column as PDF
                column_pdfs = []
                for col_idx, (col_x0, col_x1) in enumerate(columns[:5]):
                    try:
                        column_pdf_path = self.save_column_as_pdf(
                            section_doc, section_idx, col_idx, col_x0, col_x1, page_height
                        )
                        if column_pdf_path:
                            column_pdfs.append(column_pdf_path)
                            self.log(f"      Column {col_idx + 1}/5: Saved {column_pdf_path.name}")
                        else:
                            self.log(f"      Column {col_idx + 1}/5: Failed to save")
                    except Exception as e:
                        self.log(f"      Error processing column {col_idx + 1}/5: {e}")
                        continue
                
                return column_pdfs
            finally:
                if pix:
                    pix = None  # Free memory
                    
        finally:
            section_doc.close()
    
    def split_all_sections_into_columns(self) -> List[Path]:
        """Process all 15 section PDFs and split each into 5 columns."""
        self.log("="*60)
        self.log("STEP 2: Column Splitting")
        self.log("="*60)
        
        # Find all section PDFs
        section_files = sorted(self.sections_dir.glob("section_*.pdf"))
        
        if not section_files:
            self.log("Error: No section PDFs found in data/sections/")
            self.log("Please run step1_split_sections.py first")
            return []
        
        self.log(f"Found {len(section_files)} section PDFs to process")
        
        all_column_pdfs = []
        
        for section_file in section_files:
            # Extract section index from filename (section_XX.pdf -> XX)
            try:
                section_num = int(section_file.stem.split('_')[1])
                section_idx = section_num - 1  # Convert to 0-based index
            except (ValueError, IndexError):
                self.log(f"  Warning: Could not parse section number from {section_file.name}, skipping")
                continue
            
            try:
                column_pdfs = self.process_section_pdf(section_file, section_idx)
                if column_pdfs:
                    all_column_pdfs.extend(column_pdfs)
            except Exception as e:
                self.log(f"  Error processing {section_file.name}: {e}")
                import traceback
                self.log(f"  Traceback: {traceback.format_exc()}")
                continue
        
        self.log(f"\nSuccessfully created {len(all_column_pdfs)} column PDFs!")
        self.log(f"Output directory: {self.columns_dir.absolute()}")
        
        # Save log file
        log_path = self.output_dir / "step2_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        
        self.log(f"Log saved to: {log_path}")
        
        return all_column_pdfs


def main():
    """Main entry point."""
    splitter = ColumnSplitter()
    
    try:
        column_pdfs = splitter.split_all_sections_into_columns()
        print(f"\nSuccessfully split sections into {len(column_pdfs)} column PDFs!")
    except Exception as e:
        print(f"Error during column splitting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

