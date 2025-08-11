# edital_extractor.py
from __future__ import annotations
import asyncio, re, json, warnings, sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import polars as pl

import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- UTILS ----------
BRL_RE = re.compile(r"R\$\s?([\d\.]*,\d{2})")
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")

def normaliza_moeda(txt: str) -> Optional[float]:
    m = BRL_RE.search(txt)
    if m:
        try:
            return float(m.group(1).replace(".", "").replace(",", "."))
        except (ValueError, AttributeError):
            return None
    return None

def normaliza_data(txt: str) -> Optional[datetime]:
    m = DATE_RE.search(txt)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d/%m/%Y")
    except ValueError:
        return None


# ---------- EXTRAÇÃO ----------
def is_scan(path: Path) -> bool:
    """True se o PDF tiver < 50 chars de texto em todas as páginas."""
    try:
        with fitz.open(path) as doc:
            for page in doc:
                if len(page.get_text()) > 50:
                    return False
            return True
    except Exception as e:
        warnings.warn(f"Error checking if PDF is scan: {e}")
        return False


def extract_native(path: Path, edital: str) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tables = page.find_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 4,
                    }
                )
                for t in tables:
                    extracted = t.extract()
                    if not extracted:
                        continue
                    for r_idx, row in enumerate(extracted):
                        if row is None:
                            continue
                        for c_idx, cell in enumerate(row):
                            cell_str = str(cell or "").strip()
                            valor_norm = normaliza_moeda(cell_str)
                            data_norm = normaliza_data(cell_str)
                            rows.append(
                                {
                                    "edital": str(edital),
                                    "pagina": int(page.page_number),
                                    "linha": int(r_idx),
                                    "coluna": str(f"col_{c_idx}"),
                                    "valor": str(cell_str),
                                    "valor_normalizado": valor_norm,
                                    "data_abertura": data_norm,
                                }
                            )
    except Exception as e:
        warnings.warn(f"Error in native extraction for {edital}: {e}")
    
    if not rows:
        warnings.warn(f"No tables found in {edital}")
        
    # Define consistent schema
    schema = {
        "edital": pl.Utf8,
        "pagina": pl.Int32,
        "linha": pl.Int32,
        "coluna": pl.Utf8,
        "valor": pl.Utf8,
        "valor_normalizado": pl.Float64,
        "data_abertura": pl.Datetime("us"),
    }
    
    return pl.DataFrame(rows, schema=schema, strict=False)


def extract_scan(path: Path, edital: str) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []
    try:
        tables = camelot.read_pdf(str(path), pages="all", flavor="lattice", line_scale=40)
        for t in tables:
            try:
                page_num = int(t.page)
                for r_idx, row in t.df.iterrows():
                    for c_idx, cell in enumerate(row):
                        cell_str = str(cell or "").strip()
                        valor_norm = normaliza_moeda(cell_str)
                        data_norm = normaliza_data(cell_str)
                        rows.append(
                            {
                                "edital": str(edital),
                                "pagina": int(page_num),
                                "linha": int(r_idx),
                                "coluna": str(f"col_{c_idx}"),
                                "valor": str(cell_str),
                                "valor_normalizado": valor_norm,
                                "data_abertura": data_norm,
                            }
                        )
            except (ValueError, IndexError) as e:
                warnings.warn(f"Error processing table on page {t.page}: {e}")
                continue
    except Exception as e:
        warnings.warn(f"Error in scan extraction for {edital}: {e}")
    
    if not rows:
        warnings.warn(f"No tables found in scanned PDF {edital}")
        
    # Define consistent schema
    schema = {
        "edital": pl.Utf8,
        "pagina": pl.Int32,
        "linha": pl.Int32,
        "coluna": pl.Utf8,
        "valor": pl.Utf8,
        "valor_normalizado": pl.Float64,
        "data_abertura": pl.Datetime("us"),
    }
    
    return pl.DataFrame(rows, schema=schema, strict=False)


async def extract_edital(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    edital = path.stem
    try:
        return extract_scan(path, edital) if is_scan(path) else extract_native(path, edital)
    except Exception as e:
        raise RuntimeError(f"Failed to extract edital {edital}: {e}")


async def pipeline(path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = await extract_edital(path)
    outfile = out_dir / f"{path.stem}.parquet"
    df.write_parquet(outfile)
    return outfile


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse, asyncio, sys

    parser = argparse.ArgumentParser(description="Extrai tabelas de editais do governo.")
    parser.add_argument("pdf", type=Path, help="Arquivo PDF do edital")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Pasta de saída")
    args = parser.parse_args()

    try:
        outfile = asyncio.run(pipeline(args.pdf, args.out))
        print(f"Salvo: {outfile}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)