# P&ID Symbol Extraction - Three-Phase Modular Pipeline

A modular three-phase extraction pipeline for extracting P&ID (Piping and Instrumentation Diagram) symbols from PDF documents.

## Project Structure

```
Symbols/
├── src/                          # Source code directory
│   ├── step1_split_sections.py   # Phase 1: Split PDF into 15 sections
│   ├── step2_split_columns.py    # Phase 2: Split sections into 5 columns each
│   └── step3_extract_symbols.py   # Phase 3: Extract symbols from columns
├── data/                         # Data storage directory
│   ├── input/                    # Input PDFs
│   │   └── pid-legend.pdf
│   ├── sections/                 # Phase 1 output: 15 section PDFs
│   │   ├── section_01.pdf
│   │   ├── section_02.pdf
│   │   └── ... (15 files)
│   ├── columns/                  # Phase 2 output: 75 column PDFs
│   │   ├── section_01_01.pdf
│   │   ├── section_01_02.pdf
│   │   └── ... (75 files: 15 sections × 5 columns)
│   └── symbols/                  # Phase 3 output: extracted symbol images
│       ├── Pressure_Gauge.png
│       ├── Valve_3_Way.png
│       └── ... (symbol images)
├── output/                       # Final outputs
│   ├── metadata.json             # Symbol metadata
│   ├── step1_log.txt            # Phase 1 processing log
│   ├── step2_log.txt            # Phase 2 processing log
│   └── step3_log.txt            # Phase 3 processing log
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Overview

The extraction pipeline works in three independent phases:

### Phase 1: Section Splitting (`step1_split_sections.py`)
- Detects the top section (heading/description) and determines its boundary
- Detects 30 horizontal dashed/dotted lines (2 per section for 15 sections)
- Creates 15 section boundaries
- Saves each section as a separate PDF file: `section_01.pdf`, `section_02.pdf`, ..., `section_15.pdf`

### Phase 2: Column Splitting (`step2_split_columns.py`)
- Processes each of the 15 section PDFs
- Detects 4 vertical dashed lines (creating 5 columns per section)
- Saves each column as a separate PDF file: `section_XX_YY.pdf` format
- Total output: 75 column PDFs (15 sections × 5 columns)

### Phase 3: Symbol Extraction (`step3_extract_symbols.py`)
- Processes each of the 75 column PDFs
- Uses OCR (Tesseract) to extract text labels from column images
- Finds corresponding symbols on the left side of each label
- Validates symbols using connected component analysis
- Saves symbol images with ground truth labels as filenames
- Filename format: Replace spaces with underscores, preserve all numbers

## Installation

1. Install required Python dependencies:
```bash
pip install -r requirements.txt
```

Required Python packages:
- PyMuPDF (fitz) >= 1.23.0
- pdfplumber >= 0.10.0
- Pillow (PIL) >= 10.0.0
- pytesseract >= 0.3.10
- opencv-python (cv2) >= 4.8.0
- numpy >= 1.24.0

2. **Install Tesseract OCR** (required for Phase 3):

   **Windows:**
   - Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Run the installer (default location: `C:\Program Files\Tesseract-OCR`)
   - The installer should add Tesseract to your system PATH automatically
   - If not, manually add `C:\Program Files\Tesseract-OCR` to your system PATH
   - Restart your terminal/PowerShell after installation

   **macOS:**
   ```bash
   brew install tesseract
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

   **Verify installation:**
   ```bash
   tesseract --version
   ```

## Usage

### Step 1: Split PDF into Sections

1. Place your PDF file in `data/input/` directory:
   ```bash
   cp your-pid-legend.pdf data/input/pid-legend.pdf
   ```

2. Run Phase 1:
   ```bash
   python src/step1_split_sections.py
   ```

   This creates 15 section PDFs in `data/sections/`:
   - `section_01.pdf`, `section_02.pdf`, ..., `section_15.pdf`

### Step 2: Split Sections into Columns

3. Run Phase 2:
   ```bash
   python src/step2_split_columns.py
   ```

   This creates 75 column PDFs in `data/columns/`:
   - `section_01_01.pdf`, `section_01_02.pdf`, ..., `section_15_05.pdf`

### Step 3: Extract Symbols

4. Run Phase 3:
   ```bash
   python src/step3_extract_symbols.py
   ```

   This extracts symbols and saves them in `data/symbols/`:
   - Symbol images with ground truth labels as filenames
   - Metadata in `output/metadata.json`
   
   Expected output: ~500+ symbols from a standard P&ID legend PDF

## Output Format

### Section PDFs (Phase 1)
- 15 PDF files in `data/sections/`
- Naming: `section_01.pdf` through `section_15.pdf`

### Column PDFs (Phase 2)
- 75 PDF files in `data/columns/`
- Naming: `section_XX_YY.pdf` where XX is section number (01-15) and YY is column number (01-05)

### Symbol Images (Phase 3)
- PNG files in `data/symbols/`
- Filenames are ground truth labels with:
  - Spaces replaced with underscores
  - All numbers preserved
  - Invalid filename characters removed
- Examples: `Pressure_Gauge.png`, `Valve_3_Way.png`, `Temp_Indicator_100.png`

### Metadata JSON
Each entry contains:
```json
{
  "filename": "Pressure_Gauge.png",
  "original_label": "Pressure Gauge",
  "section_idx": 0,
  "column_idx": 2,
  "column_file": "section_01_03.pdf"
}
```

## Benefits of Modular Pipeline

1. **Independent Execution**: Each phase can be run separately
2. **Easy Debugging**: Inspect intermediate results (sections and columns) before extraction
3. **Error Recovery**: If Phase 3 fails, fix and rerun without redoing Phases 1-2
4. **Reusability**: Can reprocess any phase independently
5. **Clear Separation**: Each script has a single, focused responsibility

## How It Works

### Phase 1: Section Detection
- Uses computer vision (OpenCV) to detect horizontal dashed/dotted lines
- Pairs consecutive lines to form section boundaries
- Validates boundaries using detected section headers

### Phase 2: Column Detection
- Detects vertical dashed lines within each section
- Uses Hough line transform to identify 4 internal vertical lines
- Creates 5 columns: left edge to line1, line1 to line2, ..., line4 to right edge

### Phase 3: Symbol Extraction
- Column PDFs are image-only (rasterized), so OCR is required for text extraction
- Uses Tesseract OCR to extract text labels with bounding boxes from column images
- For each text label found:
  - Searches to the left of the label for the corresponding symbol
  - Uses connected component analysis to identify symbol regions
  - Filters out noise and validates symbol coherence (area, aspect ratio, pixel density)
  - Crops and cleans the symbol image (removes excess whitespace)
  - Saves with ground truth label as filename

## Filename Sanitization Rules

- **Spaces**: Replaced with underscores (`Pressure Gauge` → `Pressure_Gauge`)
- **Numbers**: Preserved (`Valve 3 Way` → `Valve_3_Way`)
- **Special Characters**: Preserved except invalid filename characters
- **Invalid Characters**: Removed (`< > : " / \ | ? *`)
- **Duplicates**: Numbered (`Pressure.png`, `Pressure_1.png`, etc.)

## Troubleshooting

### Phase 1: No sections detected
- Check that the PDF contains horizontal dashed lines separating sections
- Verify the PDF is not corrupted
- Check `output/step1_log.txt` for details

### Phase 2: Incorrect column count
- Verify section PDFs were created correctly
- Check that sections contain vertical dashed lines
- Check `output/step2_log.txt` for details

### Phase 3: Missing symbols or OCR errors
- **Tesseract OCR not found**: Ensure Tesseract is installed and in your system PATH
  - Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki
  - Verify installation: `tesseract --version`
  - Restart terminal/PowerShell after installation
- **No text objects found**: Column PDFs are image-only, so OCR is required
  - Check `output/step3_log.txt` for OCR error messages
  - Verify Tesseract is accessible: `python -c "import pytesseract; print(pytesseract.get_tesseract_version())"`
- Ensure text labels are properly extracted (check `output/step3_log.txt`)
- Verify column boundaries are correctly detected
- Check that symbols are positioned to the left of their labels
- Inspect column PDFs manually to verify layout

### Incorrect symbol-label pairing
- Symbols must be on the left side of labels
- Labels must be on the right side of symbols
- Vertical alignment is used to pair symbols with labels

## Notes

- The pipeline assumes a single-page PDF
- Each phase can be run independently
- Intermediate PDFs (sections and columns) are saved for inspection
- Processing logs are saved for each phase
- Symbol images are cleaned to remove excess whitespace
- **Phase 2 creates image-only PDFs**: Column PDFs are rasterized images, not text-based PDFs
- **Phase 3 requires OCR**: Since column PDFs are image-only, Tesseract OCR is required to extract text labels
- Typical extraction results: ~500+ symbols from a standard P&ID legend PDF

## License

This project is provided as-is for P&ID symbol extraction purposes.
