# Edital Parser - PDF Table Extractor

A robust Python tool for extracting tables from government edict PDFs (editais) with support for both native PDFs and scanned documents.

## Features

- **Dual extraction modes**: Native PDF parsing and OCR for scanned documents
- **Data normalization**: Automatic extraction and normalization of currency values (BRL) and dates
- **Schema validation**: Ensures data consistency using Pandera
- **Robust error handling**: Handles corrupted files, missing data, and edge cases
- **Async processing**: Efficient handling of large PDFs
- **Parquet output**: Optimized storage format for downstream processing

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. The following system dependencies are required:

#### Windows
```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH during installation

# Install Ghostscript (for Camelot)
# Download from: https://www.ghostscript.com/download/gsdnld.html
```

#### Linux/macOS
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr libtesseract-dev ghostscript

# macOS with Homebrew
brew install tesseract ghostscript
```

### Python Dependencies

Install the required Python packages:

```bash
pip install -r req.txt
```

## Usage

### Basic Usage

Extract tables from a single PDF:

```bash
python main.py edital.pdf
```

### Advanced Usage

Specify custom output directory:

```bash
python main.py edital.pdf --out ./my_output_folder
```

### Batch Processing

Process multiple PDFs:

```bash
# Windows PowerShell
Get-ChildItem *.pdf | ForEach-Object { python main.py $_ --out ./output }

# Linux/macOS
for pdf in *.pdf; do python main.py "$pdf" --out ./output; done
```

## Input Format

The tool accepts:
- **Native PDFs**: Text-based PDFs with selectable text
- **Scanned PDFs**: Image-based PDFs requiring OCR

## Output Format

Results are saved as Apache Parquet files with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `edital` | string | Source PDF filename |
| `pagina` | int32 | Page number (1-indexed) |
| `linha` | int32 | Row number within table |
| `coluna` | string | Column identifier (col_0, col_1, etc.) |
| `valor` | string | Raw cell text |
| `valor_normalizado` | float64 | Normalized currency value (BRL) |
| `data_abertura` | datetime | Parsed date (DD/MM/YYYY format) |

## Data Processing

### Currency Normalization
- Extracts BRL values in format `R$ 1.234,56`
- Normalizes to float: `1234.56`

### Date Parsing
- Extracts dates in format `DD/MM/YYYY`
- Converts to Python datetime objects

## Error Handling

The tool includes comprehensive error handling for:
- Missing or corrupted PDF files
- Malformed tables
- Invalid date/currency formats
- Missing dependencies
- Memory issues with large files

## Examples

### Example 1: Basic Extraction
```bash
python main.py data/editais/edital_2024_001.pdf
# Output: output/edital_2024_001.parquet
```

### Example 2: Custom Output
```bash
python main.py edital_complexo.pdf --out ./processed_data
# Output: processed_data/edital_complexo.parquet
```

### Example 3: Processing with Python
```python
import polars as pl

# Read the extracted data
df = pl.read_parquet("output/edital_2024_001.parquet")

# Filter for specific values
filtered = df.filter(
    pl.col("valor_normalizado") > 10000
)

# Group by page
by_page = df.group_by("pagina").agg([
    pl.count().alias("total_rows"),
    pl.sum("valor_normalizado").alias("total_value")
])
```

## Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Ensure Tesseract OCR is installed and in PATH
   - Verify installation: `tesseract --version`

2. **"Ghostscript not found"**
   - Install Ghostscript for Camelot table extraction
   - Restart terminal after installation

3. **"No tables found"**
   - Verify PDF contains actual tables
   - Check if PDF is heavily corrupted
   - Try adjusting table detection parameters

4. **Memory issues with large PDFs**
   - Process PDFs individually instead of in batch
   - Ensure sufficient RAM (recommend 4GB+ for large PDFs)

### Debug Mode

Enable verbose warnings by removing the warning filter:
```python
# Comment out this line in main.py:
# warnings.filterwarnings("ignore", category=UserWarning)
```

## Performance Tips

- **Native PDFs**: Much faster processing (text-based)
- **Scanned PDFs**: Slower due to OCR processing
- **Batch processing**: Use individual processing for large files
- **Memory**: Monitor RAM usage for PDFs > 100MB

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is provided as-is for educational and research purposes.