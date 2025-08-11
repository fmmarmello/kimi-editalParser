#!/usr/bin/env python3
"""
PDF Table Extractor using PyMuPDF
Extracts tables from PDF documents while intelligently ignoring header/footer regions
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFTableExtractor:
    """Extract tables from PDF documents using PyMuPDF with intelligent header/footer detection"""
    
    def __init__(self, pdf_path: str, output_dir: str = "output"):
        """
        Initialize the PDF table extractor
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output files
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate PyMuPDF version
        if not hasattr(fitz.Page, "find_tables"):
            raise RuntimeError("This PyMuPDF version does not support the table feature")
        
        # Open the document
        try:
            self.doc = fitz.open(str(self.pdf_path))
            logger.info(f"Opened PDF: {self.pdf_path} ({len(self.doc)} pages)")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
    
    def _analyze_text_distribution(self, sample_pages: int = 5) -> Tuple[float, float]:
        """
        Analyze text distribution across sample pages to identify header/footer regions
        
        Args:
            sample_pages: Number of pages to analyze
            
        Returns:
            Tuple of (header_boundary, footer_boundary) in points
        """
        if len(self.doc) == 0:
            return 50.0, 50.0
        
        # Analyze first few pages to find consistent text patterns
        pages_to_analyze = min(sample_pages, len(self.doc))
        text_positions = defaultdict(int)
        
        for page_num in range(pages_to_analyze):
            try:
                page = self.doc[page_num]
                text_blocks = page.get_text("dict")["blocks"]
                
                for block in text_blocks:
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                y_pos = span["bbox"][1]  # Top Y coordinate
                                text_positions[y_pos] += 1
            
            except Exception as e:
                logger.warning(f"Could not analyze page {page_num+1}: {e}")
        
        if not text_positions:
            return 50.0, 50.0
        
        # Find consistent text positions (potential headers/footers)
        page_height = self.doc[0].rect.height
        sorted_positions = sorted(text_positions.items())
        
        # Identify header region (top 20% of page)
        header_candidates = [(y, count) for y, count in sorted_positions if y < page_height * 0.2]
        header_boundary = 0
        
        if header_candidates:
            # Find where text density drops significantly
            max_density = max(count for _, count in header_candidates)
            threshold = max_density * 0.3
            
            for y, count in reversed(header_candidates):
                if count < threshold:
                    header_boundary = y + 10  # Add small buffer
                    break
        
        # Identify footer region (bottom 20% of page)
        footer_candidates = [(y, count) for y, count in sorted_positions if y > page_height * 0.8]
        footer_boundary = 0
        
        if footer_candidates:
            max_density = max(count for _, count in footer_candidates)
            threshold = max_density * 0.3
            
            for y, count in footer_candidates:
                if count < threshold:
                    footer_boundary = page_height - y + 10  # Add small buffer
                    break
        
        # Ensure reasonable defaults
        header_boundary = max(30.0, min(header_boundary, 100.0))
        footer_boundary = max(30.0, min(footer_boundary, 100.0))
        
        logger.debug(f"Detected header boundary: {header_boundary}, footer boundary: {footer_boundary}")
        return header_boundary, footer_boundary
    
    def _is_table_in_header_footer(self, table_bbox: tuple, page_height: float, 
                                 header_boundary: float, footer_boundary: float) -> bool:
        """
        Check if a table is in the header or footer region using adaptive boundaries
        
        Args:
            table_bbox: Table bounding box (x0, y0, x1, y1)
            page_height: Total page height
            header_boundary: Detected header boundary
            footer_boundary: Detected footer boundary
            
        Returns:
            True if table is in header/footer, False otherwise
        """
        _, y0, _, y1 = table_bbox
        
        # Check if table starts in header region
        if y0 < header_boundary:
            return True
            
        # Check if table ends in footer region
        if y1 > page_height - footer_boundary:
            return True
            
        return False
    
    def extract_tables_from_page(self, page_num: int, header_boundary: float, 
                               footer_boundary: float) -> List[pd.DataFrame]:
        """
        Extract all tables from a specific page
        
        Args:
            page_num: Page number (0-indexed)
            header_boundary: Detected header boundary
            footer_boundary: Detected footer boundary
            
        Returns:
            List of DataFrames for tables found on the page
        """
        try:
            page = self.doc[page_num]
            page_height = page.rect.height
            
            # Find all tables on the page
            tables = page.find_tables()
            
            valid_tables = []
            
            for i, table in enumerate(tables):
                # Get table bounding box
                bbox = table.bbox
                
                # Skip tables in header/footer using adaptive boundaries
                if self._is_table_in_header_footer(bbox, page_height, header_boundary, footer_boundary):
                    logger.debug(f"Skipping table {i} on page {page_num+1} (in header/footer)")
                    continue
                
                # Convert to DataFrame
                try:
                    df = table.to_pandas()
                    if not df.empty:
                        # Add metadata
                        df.attrs['page'] = page_num + 1
                        df.attrs['table_index'] = i
                        df.attrs['bbox'] = bbox
                        valid_tables.append(df)
                        logger.debug(f"Extracted table {i} from page {page_num+1}")
                except Exception as e:
                    logger.warning(f"Failed to convert table {i} on page {page_num+1}: {e}")
            
            return valid_tables
            
        except Exception as e:
            logger.error(f"Error processing page {page_num+1}: {e}")
            return []
    
    def extract_all_tables(self, max_pages: Optional[int] = None, 
                          use_adaptive_detection: bool = True) -> Dict[str, Any]:
        """
        Extract all tables from the PDF
        
        Args:
            max_pages: Maximum number of pages to process (None for all)
            use_adaptive_detection: Whether to use intelligent header/footer detection
            
        Returns:
            Dictionary containing extraction results
        """
        # Analyze document for header/footer boundaries
        if use_adaptive_detection:
            header_boundary, footer_boundary = self._analyze_text_distribution()
        else:
            header_boundary = footer_boundary = 50.0
        
        results = {
            "pdf_file": str(self.pdf_path),
            "total_pages": len(self.doc),
            "processed_pages": 0,
            "total_tables": 0,
            "header_boundary": header_boundary,
            "footer_boundary": footer_boundary,
            "tables": []
        }
        
        # Determine pages to process
        pages_to_process = min(max_pages, len(self.doc)) if max_pages else len(self.doc)
        results["processed_pages"] = pages_to_process
        
        logger.info(f"Processing {pages_to_process} pages...")
        logger.info(f"Using header boundary: {header_boundary}, footer boundary: {footer_boundary}")
        
        for page_num in range(pages_to_process):
            page_tables = self.extract_tables_from_page(
                page_num, header_boundary, footer_boundary
            )
            
            for table_df in page_tables:
                table_info = {
                    "page": table_df.attrs['page'],
                    "table_index": table_df.attrs['table_index'],
                    "bbox": table_df.attrs['bbox'],
                    "rows": len(table_df),
                    "columns": len(table_df.columns),
                    "data": table_df.fillna('').to_dict('records')
                }
                results["tables"].append(table_info)
                results["total_tables"] += 1
        
        logger.info(f"Extracted {results['total_tables']} tables from {pages_to_process} pages")
        return results
    
    def save_to_json(self, results: Dict[str, Any], output_filename: Optional[str] = None) -> str:
        """
        Save extraction results to JSON file
        
        Args:
            results: Extraction results dictionary
            output_filename: Custom output filename
            
        Returns:
            Path to the saved JSON file
        """
        if not output_filename:
            output_filename = f"{self.pdf_path.stem}_tables.json"
        
        output_path = self.output_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise
    
    def close(self):
        """Close the PDF document"""
        if hasattr(self, 'doc'):
            self.doc.close()


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Extract tables from PDF documents")
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-p", "--pages", type=int, help="Maximum pages to process")
    parser.add_argument("--fixed-margins", action="store_true", 
                       help="Use fixed margins instead of adaptive detection")
    parser.add_argument("--header-margin", type=float, default=50, 
                       help="Fixed header margin height (default: 50)")
    parser.add_argument("--footer-margin", type=float, default=50,
                       help="Fixed footer margin height (default: 50)")
    parser.add_argument("--output-file", help="Custom output filename")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = None
    try:
        extractor = PDFTableExtractor(args.pdf_file, args.output)
        results = extractor.extract_all_tables(
            max_pages=args.pages,
            use_adaptive_detection=not args.fixed_margins
        )
        
        output_path = extractor.save_to_json(results, args.output_file)
        
        # Print summary
        print(f"\nExtraction Summary:")
        print(f"PDF: {results['pdf_file']}")
        print(f"Pages processed: {results['processed_pages']}/{results['total_pages']}")
        print(f"Tables found: {results['total_tables']}")
        if not args.fixed_margins:
            print(f"Adaptive header boundary: {results['header_boundary']:.1f}")
            print(f"Adaptive footer boundary: {results['footer_boundary']:.1f}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1
    finally:
        if extractor:
            extractor.close()
    
    return 0


if __name__ == "__main__":
    exit(main())