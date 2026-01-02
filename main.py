import argparse
import os
from pathlib import Path
from pdf_to_md_textfirst import PDFtoMarkdownTextFirst, ConverterConfig
from lmstudio_client import LMStudioClient
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Text-First PDF to Markdown Converter (Local Edition)")
    
    # Default folders relative to script
    default_input = os.path.join(os.getcwd(), 'input')
    default_output = os.path.join(os.getcwd(), 'output')

    parser.add_argument('--input', '-i', default=default_input, help="Input folder containing PDFs")
    parser.add_argument('--output', '-o', default=default_output, help="Output folder for Markdown")
    parser.add_argument('--single', '-s', default=None, help="Process a single PDF file (overrides --input)")
    
    args = parser.parse_args()

    print("=== PDF to Markdown Converter (Text-First Edition) ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("======================================================")

    # Create directories
    for d in [args.input, args.output]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # Check LMStudio connection
    client = LMStudioClient()
    if not client.check_connection():
        print("ERROR: Cannot connect to LMStudio at http://localhost:1234")
        print("Please start LMStudio and load the Vision model (qwen/qwen2.5-vl-7b)")
        return

    cfg = ConverterConfig()
    converter = PDFtoMarkdownTextFirst(out_root=args.output, client=client, cfg=cfg)

    # Single file mode
    if args.single:
        pdf_path = args.single
        if not os.path.exists(pdf_path):
            print(f"ERROR: File not found: {pdf_path}")
            return
        try:
            out = converter.convert(pdf_path)
            print(f"SUCCESS: {out}")
        except Exception as e:
            print(f"FAILED: {e}")
        return

    # Batch mode
    pdf_files = sorted([f for f in os.listdir(args.input) if f.lower().endswith('.pdf')])
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(args.input, pdf_file)
        stem = Path(pdf_file).stem
        
        # Check if already processed (look for any run directory)
        existing = list(Path(args.output).glob(f"{stem}_*/{stem}.md"))
        if existing:
            print(f"Skipping {pdf_file} (already processed)")
            continue
        
        print(f"\n--- Processing: {pdf_file} ---")
        try:
            out = converter.convert(pdf_path)
            print(f"SUCCESS: {out}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\n=== All tasks completed ===")

if __name__ == "__main__":
    main()
