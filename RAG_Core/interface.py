"""RAG Backend: Two lean functions for FireGPT

This module provides two main functions for handling RAG (Retrieval-Augmented Generation) 


Functions
---------
- rag_first_turn() : Called once after routing/diagnostics to provide the first 
                    (map-based) response and initialize a fresh conversation history.
- rag_chat_turn()  : Called for every subsequent message from the chat window 
                    to maintain and update the conversation history.

Both functions block only briefly (no while-loops) and can therefore safely 
run within Dash callbacks.


Example Usage in Dash Callback
------------------------------
>>> answer, history = rag_first_turn("", diag)
>>> # ... user interaction ...
>>> answer, history = rag_chat_turn(user_msg, diag, history)
"""


from __future__ import annotations
import re
import os
from typing import List, Dict, Tuple
from langchain.schema import Document


os.environ["GGML_METAL_VERBOSE"] = "0"
from . import query_llm, context_retrieval_pipeline, query_enhancer
from utils.status_bus import set_status as set_rag_status

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv("FIREGPT_MODEL_PATH", "rag_core/local_models/gemma2.gguf")
DB_PATH: str = os.getenv("FIREGPT_DB_PATH", "rag_core/chroma_db_en")
MAX_HISTORY_LEN: int = 5  # Chat context, not retrieval context
USE_STATIC_TEST_PROMPT = False  # Set to False for production use

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------s------------------------------

def format_markdown_to_bulleted_blocks(text):
    """Format markdown text into bulleted blocks with proper structure."""
    # Remove lines like "---"
    text = re.sub(r'^\s*---\s*$', '', text, flags=re.MULTILINE)

    # Convert ## Heading to ## Heading (large) – without :
    text = re.sub(r'^\s*##\s*(.+?):?\s*$', r'## \1', text, flags=re.MULTILINE)

    # Convert "Helicopter" and other bold sections to ## Subheadings
    text = re.sub(r'^\s*\*\*(Deployed Stations|Helicopter|Additional Considerations)\*\*\s*$', r'## \1', text, flags=re.MULTILINE)

    # Replace each line that begins with "- " with bullet point (stays the same)
    # Remove leading whitespace
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)

    # Make subheadings bold (Additional Considerations)
    text = re.sub(r'(?<=\n)([-*]\s*)(\*\*[A-Za-z ,/&]+?:\*\*)', r'\1\2', text)

    # If lines begin with "**Word:**", but no bullet before → add bullet
    text = re.sub(r'^(?![-*])(\s*\*\*[A-Za-z ,/&]+?:\*\*)', r'- \1', text, flags=re.MULTILINE)

    # Separate clear paragraphs within a bullet point (e.g. with 2 spaces)
    text = re.sub(r'([^\n])\n(?=\S)', r'\1  \n', text)

    return text.strip()

def remove_triple_quotes(text: str) -> str:
    """Remove all occurrences of triple quotes (''' and ```) from the text."""
    return text.replace("'''", "").replace("```", "").replace("```markdown", "").replace("markdown", "").replace("[", "").replace("]", "")   

def _retrieve_docs(user_question: str, bundesland: str, db_path: str, diag: dict, *, model_path: str | None = None):
    """Retrieve documents based on user question and diagnostics."""
    return query_enhancer.retrieve_docs(
        user_question=user_question,
        bundesland=bundesland,
        diag=diag,
        db_path=db_path,
        model_path=model_path or MODEL_PATH,
    )

def _retrieve_docs_chat_only(user_question: str,
                             bundesland: str,
                             db_path: str,
                             model_path: str) -> List[Document]:
    """Retrieve documents for chat-only context without diagnostics."""

    refined_query = query_enhancer.optimize_user_question_for_retrieval(
        model_path=model_path,
        user_question=user_question
    )
    set_rag_status(f"Refined user query: {refined_query}")
    docs = context_retrieval_pipeline.general_context_retrieval(
        search_query=refined_query,
        bundesland=bundesland,          
        search_database=db_path,
        return_docs=True
    )
    return docs

def rag_first_turn(
    user_question: str,
    diag: Dict,
    *,
    model_path: str = MODEL_PATH,
    db_path: str = DB_PATH,
) -> Tuple[str, List[Dict]]:
    """Handle the first turn of RAG conversation with map-based response."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM model not found: {model_path}")

    print(diag)
    bundesland: str = diag.get("state", "Unknown")

    # === TESTMODE: Use static prompt and context ===
    if USE_STATIC_TEST_PROMPT:

        docs, enhanced_query = _retrieve_docs(
            user_question=user_question,
            bundesland=bundesland,
            db_path=db_path,
            diag=diag,
            model_path=model_path
        )
        if not docs:
            fallback_text = "No relevant documents found. Please adjust your query or try again."
            return fallback_text, [
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": fallback_text}
            ]
         # --- Join the summarized documents ---
        full_context = docs[0].page_content

        # Context for LLM response (overrides retrieval)

        full_context = """
        - Due to partly flat pine stocks and relatively dry soils, the areas of the forest offices Weida, Jena-Holzland, Neustadt, Saalfeld-Rudolstadt, Bad Berka and Erfurt-Willrode were classified as areas with medium forest fire risk.
- The remaining Thuringian forest offices are areas with a low risk of forest fire.
- Different factors (e.g. dead wood on the ground) pose a corresponding risk to ground fire.
- Depending on the number of forecasting regions, different forest fire hazard levels (1-5) can apply within a forest office.
- The local responsible Thuringian Forest Office provides information about its own current forest fire hazard levels in its area of responsibility.
- In the context of the task management according to § 7 para. 1 Nr. 5 ThürBKG, the Landesverwaltungsamt (Landesverwaltungsamt) pursuant to § 27 para. 2 ThürBKG as the upper civil protection authority is responsible for the civil protection of installations and dangerous events, from which risks arise for the territory of several lower civil protection authorities and which require central measures.
- The upper civil protection authority regularly merges the district descriptions to a risk assessment of the country and draws up on their basis civil protection plans of the country according to § 31 para. 2 ThürBKG in conjunction with para. 1 Nr. 2.
- In accordance with Appendix 17, a mobile civil protection management unit of the country and a wider civil protection support unit of the measurement line will be set up at the Landesfeuerwehr- und Katastropheschutzschule, as well as in accordance with Appendix 13.
- The upper civil protection authority regularly merges the district descriptions to a risk assessment of the country and draws up on their basis civil protection plans of the country according to § 31 para. 2 ThürBKG in conjunction with § 1 Nr. 2.
- On the basis of § 54 para. 1 No. 2 and para. 2 Sentence 2 of the Thuringian Fire and Civil Protection Act (ThürBKG), as amended on 5 February 2008 (GVBl. p. 22), as last amended by Article 1 of the Law of 29 June 2018 (GVBl.
- In principle, the owner of a property also has the responsibility for its traffic safety, i.e. he has to take precautions to ensure that his property does not have any effects that affect the life and limb, health, property or other rights of others.This general traffic security obligation derives from the obligation to pay damages in accordance with § 823 BGB.
- Forest fire: what to do? Legal basis for forest fire protection// Karsten Pfaue // 22. April 2021 (forest) fires are to be reported to the fire department immediately: emergency call 112 (mandatory pursuant to the German law).
- In the context of the task management according to § 7 para. 1 Nr. 5 ThürBKG, the Landesverwaltungsamt (Landesverwaltungsamt) pursuant to § 27 para. 2 ThürBKG as the upper civil protection authority is responsible for the civil protection of installations and dangerous events, from which risks arise for the territory of several lower civil protection authorities and require central measures.
- Forest fire protection Owner obligations Legal basis in forest fire protection//Karsten Pfaue // 22 April 2021 In principle, the owner of a property also has the responsibility for its traffic safety, i.e. he has to take precautions to ensure that his property does not have any effects that affect the life and limb, health, property or other rights of others.This general traffic security obligation derives from the obligation to pay damages in accordance with § 823 BGB.
- In areas at risk of forest fires, fire brigades/territory communities (e.g. with support from forest offices) should regularly carry out operational exercises in order to know o specifics and dangers in the fight against forest fires o to provide extinguishing water supply over long hose lines in inaccessible forest areas o to practice water extraction possibilities, network of paths, escape points or similar to explore
- According to the valid Thuringian Civil Protection Ordinance (ThürKatSVO), exercises are to be completed as follows for the lower disaster protection authorities: • Plan exercises and alarming exercises (annually) • Staff framework exercises (every 2 years) • Full exercises (every 5 years)
- In the regions of Mecklenburg-Vorpommern, which are at risk of forest fire, this is the task of the district-related working groups "Waldbrandschutz" (see co-operation).
- For indirect (e.g. discarding retardants as holding lines) or combined tactics (stop lines, tactical use of useful fires in combination with a soldering set) the corresponding tactical and technical requirements are currently lacking in Germany.
- Preventive discharge of extinguishing water (also with netting agents) on "green" areas for irrigation or wetting the area or vegetation with aircraft is uneconomical and should be avoided.
- The Strategic Forest Fire Protection Concept of the Free State of Saxony. Fire Protection, pp. 642-649.
- Water sampling points, good development, functioning monitoring and a well-established control team are the keys to success.
-  According to the valid Thuringian Civil Protection Ordinance (ThürKatSVO), exercises are to be completed as follows for the lower disaster protection authorities: • Plan exercises and alarming exercises (annually) • Staff framework exercises (every 2 years) • Full exercises (every 5 years)
- Helicopters are particularly suitable for: • exploration • guiding operational forces • transport of operational forces and equipment • for extinguishing fires by means of an external fire load vessel (LAB)
- Depending on the availability of helicopters from the state or federal police as well as the Bundeswehr, different long flight times can be calculated until arrival at the site.
- In the countries Baden-Württemberg, Bavaria, Hesse, Lower Saxony, North Rhine-Westphalia, Rhineland-Palatinate, Saxony-Anhalt and Thuringia, the helicopters of the state police have the possibility to accommodate external load containers for fire fighting (e.g. Bambi-Buckets).
- In the Federal Police, a large part of the helicopters are equipped with this possibility.
- The Federal Police, the Bundeswehr and private providers (air rescue, entrepreneurs) can be used.
- The share of operations in fire and civil protection of the helicopter squadrons (police of the Länder as well as the Bundespolizei) amounts to only 2.3% of the total deployment volume.
- Wind strength and wind direction as well as possible change of wind direction
- surrounding infrastructure or vegetation type (e.g. quickly combustible)
- already existing propagation/fire development and intensity of fire
- main direction of fire propagation
- special terrain conditions (e.g. slopes)
- accessibility of fire/movements, turning possibilities (also for escape routes)
- water sampling points/leakage water supply
- Fires in slope or steep slopes, as they occur in mountainous areas, are particularly critical.
- Depending on steep slopes or settlements close to the fire, farmsteads or other objects to be protected, further dangers can be identified.
- The greatest danger in the case of vegetation fires is usually the spread.
- As already described, the safety of the forces is usually the top priority in the fight against vegetation fire.
- In the event of a vegetation fire, the type of fire (e.g. ground fire, full fire, etc.)
-  A extinguisher set against the wind is only possible if the fire smoke and the thermals allow this."""

        # ---------------- Compose route/weather info ---------------
        route_info = {
            "bundesland": bundesland,
            "routes": diag.get("stations_route", []),
            "heli_route": diag.get("heli_route", []),
            "fire_radius": diag.get("fire_radius"),
        }

        print("\n=== STATIC TESTMODE ACTIVE ===\n")

    # === Use real retrieval and dynamic data ===
    else:

        # ---------------- Retrieval ----------------
    
        docs, enhanced_query = _retrieve_docs(
            user_question=user_question,
            bundesland=bundesland,
            db_path=db_path,
            diag=diag,
            model_path=model_path
        )
        if not docs:
            fallback_text = "No relevant documents found. Please adjust your query or try again."
            return fallback_text, [
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": fallback_text}
            ]
        
        # --- Join the summarized documents ---
        full_context = docs[0].page_content
        print(f"---Enhanced Query --- \n {enhanced_query}")
        print(f"\n --- Final LLM input context (first 1000 chars): --- \n{full_context[:1000]}...\n")
        
        print(f"Stations Route Initial: {diag.get('stations_route', [])}")

        # ---------------- Compose route/weather info ---------------
        route_info = {
            "bundesland": bundesland,
            "routes": diag.get("stations_route", []),
            "heli_route": diag.get("heli_route", []),
            "fire_radius": diag.get("fire_radius"),
        }
        print(f"Routes: {route_info}")



    full_context = remove_triple_quotes(full_context)

    # ---------------- LLM Call ----------------------------------------  
    answer = query_llm.query_llm_for_first_map_output(
        model_path=model_path,
        context=full_context,
        map_scenario=enhanced_query,
        route_info=route_info,
    )

    # --- preprocess answer ---
    # answer = strip_dangerous_markdown(answer)
    answer = format_markdown_to_bulleted_blocks(answer)
    answer = remove_triple_quotes(answer)
    # === Fallback if no meaningful answer ===
    if not answer or len(answer.strip()) < 10:
        answer = "No meaningful response could be generated."


    # ---------------- Prepare history -----------------------------
    history: List[Dict] = []
    if user_question.strip():
        history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": answer})

    print(f"\n=== FireGPT ===\n{answer}\n")

    return answer, history

def rag_chat_turn(
    user_question: str,
    diag: Dict,
    history: List[Dict],
    *,
    model_path: str = MODEL_PATH,
    db_path: str = DB_PATH,
    max_history_length: int = MAX_HISTORY_LEN,
) -> Tuple[str, List[Dict]]:
    """Respond to every subsequent chat message.

    Parameters
    ----------
    user_question:
        New question from the user.
    diag:
        Diagnostics (for state filter etc.).
    history:
        Previous chat history (role/content). Will be extended in-place and
        truncated to maximum `max_history_length` user/assistant pairs.
    """

    if not user_question.strip():
        raise ValueError("user_question must not be empty")

    bundesland: str = diag.get("state", "Unknown")

    # ---------------- Retrieval (question optimization only) ----------------
    docs = _retrieve_docs_chat_only(
        user_question,
        bundesland,
        db_path,
        model_path=model_path,
    )

    # --- Summarize Content---
    if docs:
        raw_context = "\n\n".join(doc.page_content for doc in docs)
        summarized_context = query_enhancer.summarize_or_select_relevant_context(
            model_path=model_path,
            context=raw_context,
            user_question=user_question,
        )
    else:
        summarized_context = ""
        print("No documents found.")


    summarized_context = remove_triple_quotes(summarized_context)
    # ---------------- LLM Call ----------------------------------------
    answer = query_llm.query_normal_chat_llm(
        model_path=model_path,
        conversation_history=history,
        current_question=user_question,
        full_context=summarized_context,
        max_history_length=max_history_length,
    )

    answer = format_markdown_to_bulleted_blocks(answer)
    print(f"GPT:{answer}")
    # ---------------- Update history ---------------------------
    # history.append({"role": "user", "content": user_question})
    # history.append({"role": "assistant", "content": answer})

    # Keep only the last max_history_length pairs (so 2x as many turns)
    turns_to_keep = max_history_length * 2
    if len(history) > turns_to_keep:
        history[:] = history[-turns_to_keep:]

    return answer, history



