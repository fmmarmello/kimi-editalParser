# pip install playwright pymupdf
# playwright install chromium     # one-time browser download

import asyncio
from pathlib import Path
import pymupdf   # only needed if you want to touch the PDF afterwards

URL = "https://www.ibge.gov.br/cidades-e-estados/mg/belo-horizonte.html"
OUT_FILE = "website.pdf"

async def web_to_pdf(url: str, output: str | Path):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()
        await page.goto(url, wait_until="networkidle")  # wait for JS
        # Optional: scroll to bottom to trigger lazy-loaded images
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)

        # Generate PDF exactly like Chrome’s “Save as PDF” dialog
        pdf_bytes = await page.pdf(
            path=output,            # write to disk immediately
            format="A4",
            margin={"top": "1cm", "bottom": "1cm",
                    "left": "1cm", "right": "1cm"},
            print_background=True   # include CSS backgrounds
        )
        await browser.close()
        return pdf_bytes           # bytes already written to disk

# ------------------------------------------------------------------
# 1. Run the browser step
asyncio.run(web_to_pdf(URL, OUT_FILE))

# 2. (Optional) post-process with PyMuPDF -----------------------------
doc = pymupdf.open(OUT_FILE)
# e.g. add page numbers:
for page in doc:
    rect = page.rect + (0, 0, 0, -20)  # bottom margin
    page.insert_text(
        pymupdf.Point(rect.width/2, rect.height + 12),
        str(page.number + 1),
        fontname="helv",
        fontsize=10,
        rotate=0,
        color=(0.4, 0.4, 0.4)
    )
doc.save("website_with_pagenumbers.pdf")
doc.close()