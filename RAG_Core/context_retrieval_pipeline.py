from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

NUM_RELEVANT_CHUNKS = 8  # Number of relevant chunks to retrieve

def general_context_retrieval(search_query: str, bundesland: str, search_database: str = "chroma_db_en",  bundesland_kategorie: str = "", return_docs: bool = False) -> str | list[Document]:
    """
    Retrieve context from a Chroma vector store based on a user question.
    Initializes the embedding model, connects to the vector store, and performs a similarity search.

    Args:
        search_query (str): The user's question.
        bundesland (str): The federal state to search in.
        search_database (str, optional): The path to the Chroma DB. Defaults to "chroma_db_en".
        bundesland_kategorie (str, optional): The state category to filter by. Defaults to "".
        return_docs (bool, optional): If True, returns a list of Document objects. Defaults to False.

    Returns:
        str | list[Document]: A formatted string of context or a list of Document objects.
    """
    print("--- Running General Retrieval ---")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    db = Chroma(persist_directory=search_database, embedding_function=embedding_model)


    all_results_with_scores = []
 

    filter = {"kategorie": {"$ne": bundesland_kategorie}} 


    # ================= 1. OR-Search: (State + Category) OR Germany-wide ====
    print(f"1. Search for: '{search_query}' for ('{bundesland}' + not Legal) OR 'Germany-wide + not Legal'")
    base_results = db.similarity_search_with_score(
        query=search_query,
        k=NUM_RELEVANT_CHUNKS,
        filter=filter
    )
    all_results_with_scores.extend(base_results)


    print(f"2. Results for OR-Filter: {len(base_results)} documents")
    for i, (doc, score) in enumerate(base_results, 1):
        pdf = doc.metadata.get("pdf_title", "Unknown PDF")
        snippet = doc.page_content[:60].replace("\n", " ")
        print(f"   {i}. PDF: {pdf} | Score: {score:.4f} | Text: {snippet}...")




    # --- Remove duplicates (based on page_content) ---
    results_by_content = {}
    for doc, score in all_results_with_scores:
        if doc.page_content not in results_by_content:
            results_by_content[doc.page_content] = doc

    deduped_docs = list(results_by_content.values())
    
    print(f"5. After duplicate removal: {len(deduped_docs)} unique documents")

    if return_docs:
        return deduped_docs

    return "\n\n---\n\n".join([doc.page_content for doc in deduped_docs])

def routed_context_retrieval_subquestion(
    search_query: str,
    bundesland: str,
    bundesland_kategorie: str,
    search_database: str = "chroma_db_en",
    return_docs: bool = False
) -> str | list[Document]:
    """
    Perform two search runs and deduplicate results.
    Used for sub-questions: (1) State + Category, (2) Germany-wide + Category.

    Args:
        search_query (str): The search query.
        bundesland (str): The federal state.
        bundesland_kategorie (str): The state category.
        search_database (str, optional): Path to Chroma DB. Defaults to "chroma_db_en".
        return_docs (bool, optional): Return documents if True. Defaults to False.

    Returns:
        str | list[Document]: Formatted string or list of Document objects.
    """
    print(f"\n--- Subquery Retrieval for: '{search_query}' ---")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    db = Chroma(persist_directory=search_database, embedding_function=embedding_model)

    def search_with_filter(bl_filter: str):
        """Helper function to search with specific filter."""
        filter = {
            "$and": [
                {"bundesland": {"$eq": bl_filter}},
                {"kategorie": {"$eq": bundesland_kategorie}}
            ]
        }
        return db.similarity_search_with_score(query=search_query, k=NUM_RELEVANT_CHUNKS, filter=filter)

    # 1. Search in federal state
    print(f"1. Search in federal state: '{bundesland}'")
    bundesland_results = search_with_filter(bundesland)

    # 2. Search Germany-wide
    print("2. Search in Germany-wide")
    deutschland_results = search_with_filter("Deutschlandweit")

    # 3. Combine & deduplicate
    combined_results = bundesland_results + deutschland_results
    unique_docs = {doc.page_content: doc for doc, score in combined_results}
    deduped = list(unique_docs.values())
    print(f"3. After deduplication: {len(deduped)} unique documents")

    if return_docs:
        return deduped
    return "\n\n---\n\n".join([doc.page_content for doc in deduped])
