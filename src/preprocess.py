# preprocess.py
import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class LaborDocumentsPreprocessor:
    def __init__(self, raw_data_dir="../data/raw", output_dir="../data/processed"):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def clean_text(self, text):
        """
        Removes irrelevant content and cleans text
        """
        # Removing extra whitespace and newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r' +', ' ', text)  # Remove multiple spaces
        text = text.strip()

        # Removing page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page \d+ of \d+', '', text)

        # Saving clean lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skipping navigation
            if len(line) < 15 and ('Labour Department' in line or 'www' in line):
                continue
            if line.isdigit():
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def load_and_clean_pdfs(self):
        """Load all PDFs -> extract and clean"""
        all_docs = []

        pdf_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file}")
            try:
                loader = PyPDFLoader(os.path.join(self.raw_data_dir, pdf_file))
                pages = loader.load()
                full_text = "\n".join([page.page_content for page in pages])
                cleaned_text = self.clean_text(full_text)
                all_docs.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": pdf_file, "type": "pdf"}
                ))
                print(f"Extracted {len(cleaned_text)} chars")
            except Exception as e:
                print(f"Error: {e}")

        return all_docs

    def load_and_clean_txt(self):
        """Loads and cleans the scraped HTML text file"""

        txt_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.txt')]
        print(f"\nFound {len(txt_files)} text files")

        all_docs =[]

        for txt_file in txt_files:
            print(f"Processing TXT: {txt_file}")
            try:
                with open(os.path.join(self.raw_data_dir, txt_file), 'r', encoding='utf-8') as f:
                    text = f.read()

                # Clean the text
                cleaned = self.clean_text(text)

                # Extract main content (skip navigation gibberish)
                # Look for content after repeated title or first meaningful paragraph
                lines = cleaned.split('\n')
                start_idx = 0
                for i, line in enumerate(lines):
                    # Find first meaningful content (questions, bullet points, or longer sentences)
                    if (line.startswith(('What', 'Where', 'How', 'You', 'Your', 'Q:', 'FAQ',
                                         'Who', 'When', 'Why', 'FDH', 'Employer', 'Wage', 'Contract')) or
                            len(line) > 80):
                        start_idx = i
                        break

                meaningful_content = '\n'.join(lines[start_idx:])

                all_docs.append(Document(
                    page_content=meaningful_content,
                    metadata={"source": txt_file, "type": "html"}
                ))
                print(f"  Extracted {len(meaningful_content)} chars")
            except Exception as e:
                print(f"  Error: {e}")

        return all_docs

    def chunk_documents(self, documents, chunk_size=1500, chunk_overlap=150):
        """Splits documents into smaller chunks for retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"\nSplit into {len(chunks)} chunks with size={chunk_size} and overlap={chunk_overlap}")

        # Save chunks to file for inspection
        output_file = os.path.join(self.output_dir, "all_chunks.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"\n{'=' * 60}\n")
                f.write(f"CHUNK {i + 1} (Source: {chunk.metadata.get('source', 'unknown')})\n")
                f.write(f"{'=' * 60}\n")
                f.write(chunk.page_content)
                f.write("\n")

        print(f"Chunks saved to: {output_file}")
        return chunks

    def add_metadata_to_chunks(self, chunks):
        """Add rich metadata to each chunk based on source"""

        metadata_rules = {
            "FDHguideEnglish.pdf": {
                "category": "FDH",
                "worker_type": "foreign_domestic_helpers",
                "keywords": ["FDH", "foreign domestic helper", "overseas", "imported", "standard employment contract", "SEC"]
            },
            "fdw_corner_webpage.txt": {
                "category": "FDH",
                "worker_type": "foreign_domestic_helpers",
                "keywords": ["FDH", "domestic helper", "rest day", "food allowance", "employment agency"]
            },
            "fdh_hire_guidebook.txt": {
                "category": "FDH",
                "worker_type": "foreign_domestic_helpers",
                "keywords": ["FDH", "employer", "minimum wage", "MAW", "accommodation", "insurance", "visa"]
            },
            "Concise Guide Employment Ordinance.pdf": {
                "category": "general_employment",
                "worker_type": "all_workers",
                "keywords": ["employment rights", "ordinance", "statutory", "employee", "employer"]
            },
            "Concise Guide Minimum Wage.pdf": {
                "category": "wage_protection",
                "worker_type": "all_workers",
                "keywords": ["minimum wage", "SMW", "hourly rate", "wage calculation"]
            },
            "CoP_EA_Eng.pdf": {
                "category": "employment_agencies",
                "worker_type": "agencies",
                "keywords": ["employment agency", "EA", "licence", "commission", "code of practice"]
            },
            "PGEA_Chapter_3.pdf": {
                "category": "employment_agencies",
                "worker_type": "agencies",
                "keywords": ["agency rules", "do's and don'ts", "compliance"]
            }
        }
        enriched_chunks = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            if source in metadata_rules:
                # Add metadata
                for key, value in metadata_rules[source].items():
                    chunk.metadata[key] = value
            enriched_chunks.append(chunk)

        return enriched_chunks

    def save_chunks_for_rag(self, chunks):
        """Saves chunks as JSON for later use in RAG pipeline"""
        import json

        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "type": chunk.metadata.get("type", "unknown"),
                "category": chunk.metadata.get("category", "unknown"),
                "worker_type": chunk.metadata.get("worker_type", "unknown"),
                "keywords": chunk.metadata.get("keywords", [])
            })

        json_path = os.path.join(self.output_dir, "processed_chunks.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        print(f"JSON saved to: {json_path}")
        return chunk_data

    def run(self):
        """Full preprocessing pipeline"""
        print("=" * 60)
        print("LABOR DOCUMENT PREPROCESSING")
        print("=" * 60)

        # Loading and cleaning all docs
        pdf_docs = self.load_and_clean_pdfs()
        txt_docs = self.load_and_clean_txt()

        # Combining all docs
        all_docs = pdf_docs + txt_docs
        print(f"\nTotal documents loaded: {len(all_docs)}")

        # Creating chunks
        chunks = self.chunk_documents(all_docs)

        # Enrich chunks with metadata
        chunks = self.add_metadata_to_chunks(chunks)

        # Saving for RAG
        self.save_chunks_for_rag(chunks)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print(f"Output saved to: {self.output_dir}")
        print("=" * 60)

        return chunks


if __name__ == "__main__":
    preprocessor = LaborDocumentsPreprocessor()
    chunks = preprocessor.run()
    # Preview
    print("\nCHUNK PREVIEW (first 3 chunks):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:300] + "...")