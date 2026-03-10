"""
RAG Core Embedding Module

This module provides comprehensive functionality for processing documents (PDF/HTML),
extracting text, splitting into chapters and chunks, translating content, and
building vector databases for retrieval-augmented generation.
"""

import os
import re
import fitz  
import nltk
import shutil
from bs4 import BeautifulSoup
from typing import List, Tuple
from transformers import MarianTokenizer, MarianMTModel, pipeline
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import pandas as pd
from transformers import AutoTokenizer

# Initialize tokenizer for German to English translation
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Download NLTK punkt tokenizer data
nltk.download("punkt")
punkt_de = nltk.data.load("tokenizers/punkt/german.pickle")

# Add custom German abbreviations for correct punctuation detection
CUSTOM_ABBREVS = {
    "z", "bzw", "ggf", "abb", "nr", "s", "ca", "u", "usw",
    "d.h.", "d. h.", "d.h", "d. h",
    "i.d.R.", "i. d. R.", "i.d.R", "i. d R.", "i. d. R."
}
punkt_de._params.abbrev_types.update(CUSTOM_ABBREVS)

# === 1. Utility Functions ===

def split_text_into_token_chunks(text: str, tokenizer, max_tokens: int = 512) -> List[str]:
    """
    Split text into token-compatible chunks for models with token limits.

    Args:
        text: Original text to be split
        tokenizer: Appropriate tokenizer (e.g., MarianTokenizer)
        max_tokens: Maximum number of tokens per chunk (default: 512 for MarianMT)

    Returns:
        List of text chunks, each containing at most max_tokens tokens
    """
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [
        tokens[i:i + max_tokens]
        for i in range(0, len(tokens), max_tokens)
    ]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


# === 2. Text Extraction Functions ===
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract plain text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text with pages separated by newlines
    """
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_text_from_html(html_path: str) -> str:
    """
    Extract only the main content from an HTML file, ignoring layout elements.
    
    Args:
        html_path: Path to the HTML file
        
    Returns:
        Cleaned text content from the HTML file
    """
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove typical non-content elements
    for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside']):
        tag.decompose()
        
    # Also remove divs/sections with typical class names
    for tag in soup.find_all(attrs={"class": re.compile(r"(menu|header|footer|nav|breadcrumb|sidebar)", re.I)}):
        tag.decompose()

    # Optional: Nur Hauptbereich (z. B. <main>)
    main = soup.find("main")
    if main:
        soup = main

    # Clean and return text
    lines = [line.strip() for line in soup.get_text().splitlines()]
    return '\n'.join(line for line in lines if line)

def preprocess_documents_folder(folder_path: str = "Documents"):
    """
    Preprocess documents in a folder by extracting text from PDF/HTML files.
    
    Args:
        folder_path: Path to the folder containing documents to process
    """
    os.makedirs("raw_files", exist_ok=True)
    for file in os.listdir(folder_path):
        input_path = os.path.join(folder_path, file)
        base_name, ext = os.path.splitext(file)

        if ext.lower() not in (".pdf", ".html", ".htm"):
            print(f"Ignoring file: {file}")
            continue

        # Prepare target filename
        clean_title = slugify(base_name)
        output_path = os.path.join("raw_files", clean_title + ".txt")

        # Read file content
        if ext.lower() == ".pdf":
            text = extract_text_from_pdf(input_path)
        else:
            text = extract_text_from_html(input_path)

        # Save file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {os.path.basename(output_path)}")
        
# === 3. Chapter Processing Functions ===
def split_text_by_chapters(text: str) -> List[str]:
    """
    Split raw text into chapters based on numbered headings.
    If no numbered headings are found, returns the entire text as one chapter.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        List of chapter texts, each starting with its heading
    """
    # Search for numbered headings
    pattern = r"(?:^|\n)(\d+\.(?:\d+(?:\.\d+)?)?)\s+(.+?)(?=\n|$)"
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    
    if not matches:
        # No numbered headings found -> return entire text as one chapter
        print("   No numbered headings found - treating entire text as one chapter")
        return [text.strip()]
    
    chapters = []
    
    # First chapter: From text beginning to first heading
    first_match_start = matches[0].start()
    if first_match_start > 0:
        intro_text = text[:first_match_start].strip()
        if intro_text:  # Only add if not empty
            chapters.append(intro_text)
    
    # All other chapters: From heading to heading
    for i, match in enumerate(matches):
        start = match.start()
        if i + 1 < len(matches):
            # Until next heading
            end = matches[i + 1].start()
        else:
            # Last chapter: until text end
            end = len(text)
        
        chapter_text = text[start:end].strip()
        if chapter_text:  # Only add if not empty
            chapters.append(chapter_text)
    
    return chapters


# === 4. Helper Functions ===

def slugify(text: str) -> str:
    """
    Convert a title into a clean filename:
    - Replace umlauts (ä → ae, ß → ss, ...)
    - Replace spaces and hyphens with underscores
    - Remove special characters
    - Ensure filename doesn't start with a number
    
    Args:
        text: Original text to be converted
        
    Returns:
        Clean filename string
    """
    text = text.strip().lower()

    # Replace umlauts and ß
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'ß': 'ss'
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # Replace spaces and hyphens with underscores
    text = text.replace("-", "_").replace(" ", "_")

    # Keep only letters, numbers and underscores
    text = re.sub(r'[^\w_]', '', text)

    # Reduce double underscores
    text = re.sub(r'_+', '_', text)

    # Remove leading/trailing underscores
    text = text.strip('_')

    # If starting with number, add prefix
    if re.match(r'^\d', text):
        text = "doc_" + text

    return text[:60]
 
def load_global_metadata(excel_path: str = "metadata.xlsx") -> dict:
    """
    Load global metadata from an Excel file.
    
    Args:
        excel_path: Path to the Excel metadata file
        
    Returns:
        Dictionary mapping document slugs to their metadata
    """
    df = pd.read_excel(excel_path)

    meta_dict = {}

    for _, row in df.iterrows():
        raw_filename = str(row["Dateiname"]).strip()
        base, _ = os.path.splitext(raw_filename)
        slug = slugify(base)

        meta_dict[slug] = {
            "quelle": row.get("Quelle", ""),
            "format": row.get("Format", ""),
            "link": row.get("Link", ""),
            "bundesland": row.get("Bundesland", ""),
            "kategorie": row.get("Kategorien", "")
        }

    return meta_dict
    
def normalize_punctuation_spacing(text: str) -> str:
    """
    Add space after periods without spacing.
    
    Example: 'Abb.ggf.' -> 'Abb. ggf.' (important for tokenizer)
    
    Args:
        text: Text with potential punctuation spacing issues
        
    Returns:
        Text with normalized spacing after periods
    """
    return re.sub(r'\.(?=[A-Za-zÄÖÜäöüß])', '. ', text)

def fire_sent_tokenize(text: str) -> List[str]:
    """
    Sentence segmentation with enhanced punkt tokenizer.
    
    Args:
        text: Text to tokenize into sentences
        
    Returns:
        List of sentences
    """
    return punkt_de.tokenize(text.strip())

# === 5. Text Processing Functions ===

def split_chapters_into_subchunks(      
    chapters: List[str],
    sentences_per_chunk: int = 2,
    global_meta: dict = None,
    doc_key: str = None
) -> Tuple[List[str], List[Document]]:
    """
    Split chapters into overlapping sentence chunks with metadata.
    
    Process:
    - First line of chapter = title
    - Rest = body -> split into overlapping sentence windows
    - Create LangChain Document objects with metadata
    
    Args:
        chapters: List of chapter texts
        sentences_per_chunk: Number of sentences per chunk
        global_meta: Global metadata dictionary
        doc_key: Document key for metadata lookup
        
    Returns:
        Tuple of (raw text chunks, metadata list)
    """
    sub_texts, sub_metadata = [], []

    # Get global metadata for this document
    global_metadata = global_meta.get(doc_key, {}) if global_meta else {}

    for chap_idx, chap in enumerate(chapters, start=1):
        lines = chap.splitlines()
        chapter_title = lines[0].strip()
        body = normalize_punctuation_spacing(" ".join(lines[1:]).strip())

        # Get clean sentence list
        sents = fire_sent_tokenize(body)

        # Create sliding window chunks
        if len(sents) <= sentences_per_chunk:
            windows = [" ".join(sents)]
        else:
            windows = [
                " ".join(sents[i : i + sentences_per_chunk])
                for i in range(len(sents) - sentences_per_chunk + 1)
            ]

        # Collect texts and metadata only
        for sub_idx, txt in enumerate(windows, start=1):
            sub_texts.append(txt)
            sub_metadata.append({
                "pdf_title": doc_key,
                "chapter_title": chapter_title,
                "chapter_id":  chap_idx,
                "subchunk_id": sub_idx,
                **global_metadata
            })

    return sub_texts, sub_metadata


# === 6. Export Functions ===
def write_chapters_with_subchunks_to_markdown(
    sub_texts: List[str],
    meta: List[dict],
    output_dir: str
) -> None:
    """
    Write chunks to Markdown files organized by chapter.
    
    Args:
        sub_texts: List of text chunks
        meta: List of metadata dictionaries
        output_dir: Directory to write Markdown files
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Group chunks by chapter
    grouped = {}
    for txt, meta in zip(sub_texts, meta):
        cid = meta["chapter_id"]
        chapter_title = meta["chapter_title"]
        grouped.setdefault(cid, {"chapter_title": chapter_title, "chunks": []})
        grouped[cid]["chunks"].append(txt)

    # Write one file per chapter
    for cid, info in grouped.items():
        safe = "".join(c if c.isalnum() else "_" for c in info["chapter_title"])[:30]
        path = os.path.join(output_dir, f"{cid:02d}_{safe}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {info['chapter_title']}\n\n")
            for chunk in info["chunks"]:
                f.write(f"- {chunk}\n\n")

# === 7. Vector Database Functions ===
def build_subchunks_chroma_db(
    sub_docs: List[Document],
    persist_path: str,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
) -> Chroma:
    """
    Build or update a Chroma vector database from document chunks.
    
    Args:
        sub_docs: List of Document objects to embed
        persist_path: Directory to store the vector database
        embedding_model_name: HuggingFace embedding model to use
        
    Returns:
        Chroma vector database instance
    """
    embed = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Check if database already exists
    if os.path.exists(persist_path) and os.listdir(persist_path):
        # Load existing database
        print("   Loading existing vector database...")
        db = Chroma(persist_directory=persist_path, embedding_function=embed)
        
        # Add new documents
        if sub_docs:
            print(f"   Adding {len(sub_docs)} new documents...")
            db.add_documents(sub_docs)
        else:
            print("   No new documents to add.")
        
        return db
    else:
        # Create new database
        print("   Creating new vector database...")
        os.makedirs(persist_path, exist_ok=True)
        return Chroma.from_documents(
            documents=sub_docs,
            embedding=embed,
            persist_directory=persist_path
        )

# === 8. Translation Functions ===
_TRANSLATOR = None

def get_translator(model_name="Helsinki-NLP/opus-mt-de-en"):
    """
    Get or initialize a translation pipeline (singleton pattern).
    
    Args:
        model_name: HuggingFace translation model name
        
    Returns:
        Translation pipeline instance
    """
    global _TRANSLATOR
    if _TRANSLATOR is None:
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name)
        _TRANSLATOR = pipeline(
            "translation",
            model=mdl,
            tokenizer=tok,
            framework="pt",
            device=-1,
            batch_size=1
        )
    return _TRANSLATOR

def translate_subchunks(
    texts: List[str],
    model_name: str = "Helsinki-NLP/opus-mt-de-en"
) -> List[str]:
    """
    Translate texts, even if they are longer than 512 tokens.
    
    Args:
        texts: List of text chunks to translate
        model_name: HuggingFace model name
        
    Returns:
        List of translated texts
    """
    translator = get_translator(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    translations: List[str] = []
    total = len(texts)

    for idx, txt in enumerate(texts, start=1):
        perc = idx / total * 100
        print(f"   Translating chunk {idx}/{total} ({perc:.0f}%)...", end="\r")

        # Check length
        if len(tokenizer.encode(txt, truncation=False)) > 512:
            print(f"Warning: Chunk {idx} is too long - will be split.")
            sub_chunks = split_text_into_token_chunks(txt, tokenizer)
            sub_translations = [translator(chunk, max_length=512, truncation=True)[0]["translation_text"] for chunk in sub_chunks]
            full_translation = " ".join(sub_translations)
        else:
            result = translator(txt, max_length=512, truncation=True)
            full_translation = result[0]["translation_text"]

        translations.append(full_translation)

    print()
    return translations


# === 9. Main Execution ===
if __name__ == "__main__":
    # Step 1: Generate raw texts from PDF/HTML
    print("Extracting text from PDF/HTML to raw_files/ ...")
    preprocess_documents_folder("Documents")

    print("Scanning raw_files/ for .txt documents ...")
    raw_folder = "raw_files"
    raw_files = [
        f for f in os.listdir(raw_folder)
        if f.endswith(".txt") and os.path.isfile(os.path.join(raw_folder, f))
    ]
    print(f"Found {len(raw_files)} text files.\n")

    # Step 2: Load metadata
    print("Loading metadata from metadata.xlsx ...")
    global_meta = load_global_metadata("metadata.xlsx")

    all_sub_texts = []
    all_sub_docs = []
    
    print("\nMetadata verification for all documents:")
    missing_keys = []

    for filename in raw_files:
        doc_key = os.path.splitext(filename)[0]
        if doc_key not in global_meta:
            missing_keys.append(doc_key)

    if missing_keys:
        print(f"{len(missing_keys)} document(s) without metadata:")
        for key in missing_keys:
            print(f"   - {key}")
    else:
        print("All documents have matching metadata.")

    for filename in raw_files:
        doc_key = os.path.splitext(filename)[0]
        
        # Check if already processed - if yes, skip
        if os.path.exists(f"chapters_md_en/{doc_key}"):
            print(f"Skipping already processed document: {filename}")
            continue
            
        print(f"\nProcessing: {filename}")
        path = os.path.join(raw_folder, filename)

        try:
            print("   1. Loading text ...")
            raw = extract_text_from_pdf(path) if filename.endswith(".pdf") else open(path, encoding="utf-8").read()
            print("   Text loaded.")

            print("   2. Finding chapters ...")
            chapters = split_text_by_chapters(raw)
            print(f"   {len(chapters)} chapters found.")

            print("   3. Creating subchunks ...")
            sub_texts, sub_meta = split_chapters_into_subchunks(
                chapters,
                sentences_per_chunk=2,
                global_meta=global_meta,
                doc_key=doc_key
            )
            print(f"   {len(sub_meta)} subchunks created.")

            print("   4. Saving German markdown files ...")
            write_chapters_with_subchunks_to_markdown(sub_texts, sub_meta, f"chapters_md_de/{doc_key}")
            print(f"   Saved to: chapters_md_de/{doc_key}")

            # Step 6: Translation to English
            print("Translating subchunks to English ...")
            texts_en = translate_subchunks(sub_texts)
            print(f"   {len(texts_en)} translations completed.")

            print("   5. Saving English markdown files ...")
            write_chapters_with_subchunks_to_markdown(texts_en, sub_meta, f"chapters_md_en/{doc_key}")
            print("   English markdown saved.\n")

            all_sub_texts.extend(texts_en)
            all_sub_docs.extend(sub_meta)

        except Exception as e:
            print(f"Error with {filename}: {e}")
            continue

    # Step 9: English vector database
    if all_sub_texts:
        docs_en = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(all_sub_texts, all_sub_docs)
        ]

        print("Creating English vector database ...")
        db_en = build_subchunks_chroma_db(docs_en, "chroma_db_en")
        print("   English vector database saved.\n")
    else:
        print("No new documents processed - database remains unchanged.\n")

    print("\nPipeline completed successfully!")


