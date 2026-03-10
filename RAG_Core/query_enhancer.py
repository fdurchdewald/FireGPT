from typing import List, Tuple
from langchain.docstore.document import Document
from . import context_retrieval_pipeline
import re, time, threading
from utils.status_bus import set_status as set_rag_status

MAX_CONTEXT_LENGTH = 8192
subquery_enhancer = True

def clean_context(raw_context: str) -> str:
    """Remove QA/explanation blocks and markdown-like structures."""
    # Remove QA/explanation blocks and markdown-like structures
    patterns = [
        r"(?i)please provide.*?(?=\n|$)",
        r"(?i)\*\*?(answer|note|explanation)s?:?\*\*?",
        r"(?i)let me know.*?(?=\n|$)",
        r"(?i)^-{3,}$",  # "---" separators
    ]
    for pattern in patterns:
        raw_context = re.sub(pattern, "", raw_context, flags=re.MULTILINE)
    return raw_context.strip()

def delayed_status():
    """Set status message after delay."""
    time.sleep(3)
    set_rag_status("---- Still summarising retrieved content to reduce token count ... ----")


def sanitize_context(text: str) -> str:
    """Clean text by removing markdown patterns and formatting."""
    import re
    patterns = [
        r"^\s*[-–—]{3,}\s*$",          # "---", "–––"
        r"^\s*[\*]{3,}\s*$",           # "***"
        r"^\s*[\*]{2,}\s*$",           # "**"
        r"^\s*[\*]{1,}\s*$",           # "*"
        r"^\s*'{3,}\s*$",              # "'''"
        r"^\s*`{3,}\s*$",              # "```"
        r"^\s*#{1,6}\s+.*",            # Markdown headings
        r"^\s*>.*",                    # Markdown blockquotes
        r"^\s*\*\*.*\*\*\s*$",         # Markdown bold lines
        r"^\s*<[^>]+>\s*$",            # HTML tags
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n+", "\n", text)  # Reduce double empty lines
    return text.strip()


def build_enhanced_query_prompt(user_question: str, diag: dict) -> str:
    """Build enhanced query from user question and diagnostics."""
    # Mapping German federal states to English
    state_map = {
        "Bayern": "Bavaria",
        "Nordrhein-Westfalen": "North Rhine-Westphalia",
        "Niedersachsen": "Lower Saxony",
        "Baden-Württemberg": "Baden-Wuerttemberg",
        "Hessen": "Hesse",
        "Sachsen": "Saxony",
        "Rheinland-Pfalz": "Rhineland-Palatinate",
        "Berlin": "Berlin",
        "Brandenburg": "Brandenburg",
        "Sachsen-Anhalt": "Saxony-Anhalt",
        "Schleswig-Holstein": "Schleswig-Holstein",
        "Thüringen": "Thuringia",
        "Hamburg": "Hamburg",
        "Mecklenburg-Vorpommern": "Mecklenburg-Western Pomerania",
        "Bremen": "Bremen",
        "Saarland": "Saarland"
    }

    # Translate federal state
    raw_state = diag.get("state", "Unknown federal state")
    state = state_map.get(raw_state, raw_state)

    clc = diag.get("clc", {}).get("name", "unknown land cover")
    fire_risk = diag.get("fire_risk", "n/a")
    fire_radius = diag.get("fire_radius", None)

    # Convert temperature to qualitative description
    temp_val = diag.get("weather_now", {}).get("temperature_C", None)
    if temp_val is None:
        temp_desc = "unknown temperature"
    elif temp_val < 5:
        temp_desc = "very cold"
    elif temp_val < 15:
        temp_desc = "cool"
    elif temp_val < 25:
        temp_desc = "moderate temperature"
    elif temp_val < 35:
        temp_desc = "hot"
    else:
        temp_desc = "very hot"

    # Convert humidity to qualitative description
    humidity_val = diag.get("weather_now", {}).get("rel_humidity_%", None)
    if humidity_val is None:
        humidity_desc = "unknown humidity"
    elif humidity_val < 30:
        humidity_desc = "very dry air"
    elif humidity_val < 50:
        humidity_desc = "dry air"
    elif humidity_val < 70:
        humidity_desc = "moderate humidity"
    else:
        humidity_desc = "very humid conditions"

    # Count unit types (e.g., volunteer fire department, professional fire department, main fire station)
    stations = diag.get("stations_route", [])
    type_counts = {
        "Volunteer fire department": 0,
        "Professional fire department": 0,
        "Main fire station": 0
    }

    for s in stations:
        name = s.get("name", "").lower()
        if "freiwillige feuerwehr" in name:
            type_counts["Volunteer fire department"] += 1
        elif "berufsfeuerwehr" in name:
            type_counts["Professional fire department"] += 1
        elif "hauptfeuerwache" in name or "wache" in name:
            type_counts["Main fire station"] += 1

    station_desc_parts = []
    for key, count in type_counts.items():
        if count > 0:
            # Plural-Form korrekt machen
            if count == 1:
                station_desc_parts.append(f"1 {key}")
            else:
                station_desc_parts.append(f"{count} {key}s")

    station_summary = ", ".join(station_desc_parts) if station_desc_parts else "No nearby fire stations"

    # Helicopter
    heli = "available" if diag.get("heli") else "not available"

    # Water source
    water_dist = diag.get("water", {}).get("distance_m", None)
    if water_dist is None:
        water_info = "No water source nearby."
    elif water_dist < 500:
        water_info = "Water source very close."
    elif water_dist < 2000:
        water_info = "Water source nearby."
    elif water_dist < 5000:
        water_info = "Water source moderately far."
    else:
        water_info = f"Water source far away."

    # Legal notice
    legal_note = (
        f"Also consider legal regulations in {state}, such as reporting duties, "
        "responsibilities, authority escalation, and securing zones."
    )

    # Summary
    scenario_info = (
        f"Scenario in {state}, land cover: '{clc}', fire risk: '{fire_risk}', "
        f"{temp_desc}, {humidity_desc}. "
        f"Nearby units: {station_summary}. Helicopter: {heli}. "
        f"{water_info} {legal_note}."
        f"fire radius: {fire_radius} meters."
    )
    print(f"Additional information: {user_question.strip()}, Scenario: {scenario_info}")
    print(f"Stations Route Initial: {diag}")
    return f"Scenario: {scenario_info}, Additional information: {user_question.strip()}"

def generate_structured_subqueries(enhanced_query: str) -> List[Tuple[str, str]]:
    """Generate structured subqueries for different categories."""
    return [
        (
            f"Sub-question: What are the legal reporting obligations, authority structures, and responsibilities that apply in this wildfire scenario?\nScenario: {enhanced_query}",
            "Legal"
        ),
        (
            f"Sub-question: What firefighting tactics or suppression strategies are appropriate for wildfires occurring in this land cover type and under the given weather and fire risk conditions?\nScenario: {enhanced_query}",
            "Wildfire_Knowledge"
        ),
        (
            f"Sub-question: What firefighting resources, such as fire stations, vehicles, helicopters, or special units, are typically needed in scenarios like this?\nScenario: {enhanced_query}",
            "Wildfire_Knowledge"
        ),
        (
            f"Sub-question: What critical hazards such as wind, terrain, smoke, or nearby population must be considered in this wildfire scenario, and how can they be mitigated?\nScenario: {enhanced_query}",
            "Wildfire_Knowledge"
        )
    ]


def optimize_enhanced_query(model_path: str, enriched_input: str) -> str:
    """
    Use the LLM to rewrite the enhanced query into a more specific and retrieval-optimized formulation.
    """
    from llama_cpp import Llama
    print("Optimizing enhanced query via LLM...")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=MAX_CONTEXT_LENGTH,
        verbose=False
    )

    prompt = f"""
    Rewrite the following wildfire scenario as a single, complete English sentence suitable as a search query.

    - Include all facts as natural language: region, vegetation, fire radius, fire risk, weather, resources, legal duties.
    - Do NOT add or explain anything.
    - Do NOT include labels or formatting.
    - Only return one fluent sentence.

    Example Input:
    A wildfire scenario in Bavaria involving a broad-leaved forest, with a fire radius of 70 meters, high fire risk, hot temperatures, and very dry air. Two volunteer fire departments and one main fire station are nearby. A helicopter is available and a water source is close. Legal regulations in Bavaria apply, including reporting duties, responsibilities, authority escalation, and the securing of zones in case of civilian danger.

    Example Output:
    A wildfire in Bavaria affecting a broad-leaved forest under high fire risk with hot and very dry weather conditions, a fire radius of 70 meters, two nearby volunteer fire departments, one main fire station, helicopter assistance available, a nearby water source, and applicable legal duties such as reporting responsibilities, authority escalation, and securing safety zones.
    
    Input:
    {enriched_input}

    Output:
    """

    response = llm(prompt, max_tokens=250, stop=["\n", "Input:", "Output:", "Example"], echo=False, stream=False)
    return response["choices"][0]["text"].strip() if "choices" in response else enriched_input


def optimize_subquery(model_path: str, question: str) -> str:
    """
    Refine a structured subquestion using the LLM to improve retrieval specificity.
    """
    from llama_cpp import Llama
    print(f"Refining subquery: {question}")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=MAX_CONTEXT_LENGTH,
        verbose=False
    )


    prompt = f"""
    You are an experienced assistant helping to rewrite sub-questions for wildfire-related information retrieval.

    Instructions:
    - Rewrite the sub-question as one clear and specific English sentence
    - Only include information from the scenario that is clearly relevant
    - Do NOT include explanations, formatting, or placeholders

    Example Input:
    Sub-question: What legal duties and reporting requirements apply in this wildfire scenario?
    Scenario: A wildfire in Bavaria affecting a broad-leaved forest under high fire risk with hot and very dry conditions, a fire radius of 70 meters, two nearby volunteer fire departments, helicopter support, and legal regulations including reporting, authority escalation, and zone security.

    Example Output:
    What legal duties and reporting requirements apply in Bavaria during a high-risk wildfire affecting a broad-leaved forest?

    Input:
    {question}

    Output:
    """
    response = llm(prompt, max_tokens=128, stop=["\n", "Input:", "Output:", "Example"], echo=False, stream=False)
    return response["choices"][0]["text"].strip() if "choices" in response else question


def optimize_user_question_for_retrieval(model_path: str, user_question: str) -> str:
    """
    Reformulate a user's raw question into a more precise and retrieval-friendly search query,
    without using any scenario metadata.
    """
    from llama_cpp import Llama
    print("Optimizing user question for retrieval...")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=MAX_CONTEXT_LENGTH,
        verbose=False
    )

    prompt = f"""
    You are an assistant specialized in wildfire-related information retrieval.

    You receive a general user input question, which may be vague or broad.

    Your task:
    - Reformulate the question into a **concise, clear, and specific** search query
    - Keep it in natural English, suitable for querying a legal or tactical wildfire document database
    - Do not add scenario context, metadata, or invented facts
    - Only rephrase and sharpen the question
    - Output one query sentence only, with no extra formatting or explanation

    ---

    Example Input: 
    "What should we do now?"

    Example Output: 
    What are the recommended immediate actions for wildfire response?

    ---

    Input:  
    {user_question}

    Output:
    """

    response = llm(prompt, max_tokens=512, stop=["\n", "Input:", "Output:", "Example"], echo=False, stream=False)
    return response["choices"][0]["text"].strip() if "choices" in response else user_question



def retrieve_docs(user_question: str, bundesland: str, diag: dict, db_path: str, model_path: str) -> List[Document]:
    """
    Enhanced retrieval pipeline using an enriched main query and 4 structured subqueries.
    Summarizes context chunks per subquery using LLM and returns a List[Document].
    """
    print("=== Running enhanced FireGPT document retrieval ===")
    all_docs = []

    # 1. Build enhanced master query from diagnostics
    base_query = build_enhanced_query_prompt(user_question, diag)
    enhanced_query = optimize_enhanced_query(model_path, base_query)
    print(f"\n Optimized Retrieval Query: {enhanced_query} ")
    set_rag_status(f"–--- Retrieving context from all documents ----\nMaster Query: {enhanced_query} ")


    master_docs = context_retrieval_pipeline.general_context_retrieval(
    search_query=enhanced_query,
    bundesland=bundesland,
    search_database=db_path,
    return_docs=True
    )
    

    if master_docs:
        master_context = "\n\n".join([doc.page_content for doc in master_docs])
        threading.Thread(target=delayed_status).start()
        summary_text = summarize_or_select_relevant_context(model_path, master_context, user_question)
        all_docs.append(Document(page_content=summary_text))
    

    if subquery_enhancer == True:
        # 3. Run structured subqueries with categories
        subqueries = generate_structured_subqueries(enhanced_query)
        for raw_subquery, category in subqueries:
            subquery = optimize_subquery(model_path, raw_subquery)
            print(f"\n Subquery: {subquery}")
            if category == 'legal':
                set_rag_status(f"–--– Retrieving legal context from region-specific and national documents ----\nSubquery: {subquery}")
            else: 
                set_rag_status(f"–--– Retrieving tactical wildfire context from region-specific and national documents ----\nSubquery: {subquery}")
            results = context_retrieval_pipeline.routed_context_retrieval_subquestion(
                search_query=subquery,
                bundesland=bundesland,
                bundesland_kategorie=category,
                search_database=db_path,
                return_docs=True
            )
    
            if results:
                raw_text = "\n\n".join([doc.page_content for doc in results])
                threading.Thread(target=delayed_status).start()
                summary_text = summarize_or_select_relevant_context(model_path, raw_text, subquery)
                all_docs.append(Document(page_content=summary_text))
                
    
    combined_text = "\n".join([doc.page_content for doc in all_docs])

    lines = [line.strip() for line in combined_text.splitlines() if line.strip()]
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    final_cleaned_text = sanitize_context(clean_context("\n".join(unique_lines)))



    print(f"\n Final cleaned result has {len(final_cleaned_text.splitlines())} lines.")
    print(f"Context length (characters): {len(final_cleaned_text)}\n")
    return [Document(page_content=final_cleaned_text)], enhanced_query

def summarize_or_select_relevant_context(
    model_path: str,
    context: str,
    user_question: str,
) -> str:
    """
    Extract only the most relevant original text segments from retrieved context.
    """
    from llama_cpp import Llama

    print(f"\n--- 🔍 Summarizing context ---")
    print(f"Question: {user_question.strip()[:120]}{'...' if len(user_question.strip()) > 120 else ''}")
    print(f"Context length (characters): {len(context)}\n")
    context = clean_context(context)

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=MAX_CONTEXT_LENGTH,
            verbose=False
        )
        print(" LLM loaded successfully.")
    except Exception as e:
        print(f" LLM initialization failed: {e}")
        return context


    prompt = f"""
    You are an expert assistant. Extract only the original sentences from the following document that are clearly relevant to the user's question.

    Question:
    {user_question}

    Document:
    {context}

    Instructions:
    - Do NOT add explanations, answers, notes, rephrase, or summarize
    - Only include original sentences that are directly relevant for the question, or that provide clearly helpful factual, technical, practical or legal context.
    - Do NOT include questions unless they provide essential context.
    - Each relevant sentence must be on its own line, starting with "- ".

    
    Relevant Sentences:
    """

    try:
        response = llm(
            prompt,
            max_tokens=2000,
            stop=["\n---", "\nContext:", "<|END|>", "**STOP**"],
            echo=False,
            stream=False
        )
        if isinstance(response, dict) and response.get("choices"):
            summarized = response["choices"][0]["text"].strip()
            if summarized:
                print(" Summarization successful.")
                print(f"Context length (characters): {len(summarized)}\n")
                print("\n Extracted Summary:")
                print(summarized[:1000] + ("..." if len(summarized) > 1000 else ""))

                print("\n Original Context Snippet:")
                print(context[:1000] + ("..." if len(context) > 1000 else ""))

                # Return multi-line, cleaned output
                return "\n".join(
                    line.strip() for line in summarized.splitlines() if line.strip()
                )
    except Exception as e:
        print(f" Summarization failed during inference: {e}")

    print("  Returning original context as fallback.")
    return context