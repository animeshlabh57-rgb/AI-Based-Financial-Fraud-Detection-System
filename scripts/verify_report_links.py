import sys
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("BCSE497J_Project_Report_Financial_Fraud_Detection.pdf")
if not pdf_path.exists():
    print(f"Error: {pdf_path} does not exist.")
    sys.exit(1)

reader = PdfReader(str(pdf_path))
num_pages = len(reader.pages)
print(f"Total Physical Pages: {num_pages}")

total_links = 0
page_link_counts = {}
destinations = {}

# Check named destinations
try:
    named_dests = reader.named_destinations
    print(f"Named Destinations count: {len(named_dests)}")
    sample_dests = list(named_dests.keys())[:15]
    print(f"Sample destinations: {sample_dests}")
except Exception as e:
    print(f"Could not read named destinations: {e}")

# Inspect pages for Link annotations
for i, page in enumerate(reader.pages):
    phys_pg = i + 1
    if "/Annots" in page:
        annots = page["/Annots"]
        link_count = 0
        for a in annots:
            annot_obj = a.get_object()
            if annot_obj.get("/Subtype") == "/Link":
                link_count += 1
                total_links += 1
        if link_count > 0:
            page_link_counts[phys_pg] = link_count

print(f"\nTotal Link Annotations across document: {total_links}")
print("Pages with clickable links:")
for pg, cnt in page_link_counts.items():
    print(f"  Page {pg}: {cnt} clickable links")

# Check pages 12, 13 (Chapter 1) and 14 (Chapter 2)
print("\nPage allocation check:")
print(f"Physical Page 12 (Arabic 1): {page_link_counts.get(12, 0)} links")
print(f"Physical Page 13 (Arabic 2): {page_link_counts.get(13, 0)} links")
print(f"Physical Page 14 (Arabic 3): {page_link_counts.get(14, 0)} links")
print(f"Physical Page 40/41 (References): {page_link_counts.get(40, 0)} / {page_link_counts.get(41, 0)} links")
