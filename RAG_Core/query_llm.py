import os
os.environ["GGML_METAL_VERBOSE"] = "0"
os.environ["GGML_METAL_DEBUG"] = "0"
os.environ["GGML_CUDA_DEBUG"] = "0"
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
import logging
logging.getLogger("llama_cpp").setLevel(logging.ERROR)
from llama_cpp import Llama
import sys
from contextlib import contextmanager
from utils.status_bus import set_status as set_rag_status

@contextmanager
def suppress_stdout_stderr():
    """Temporarily suppress all stdout/stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _format_route_info(route_info: dict) -> str:
    """Format route information for display."""
    routes = route_info.get("routes", [])
    heli = route_info.get("heli_route", {})

    out = []

    if not routes:
        out.append("No stations available.\n")

    for i, r in enumerate(routes, 1):
        name = r.get("name", "Unknown Station")
        eta  = r.get("total_time_min", "?")
        out.append(f"{i}: {name} – {eta} min")

    if heli and isinstance(heli, dict):
        heli_min = round(heli.get("total_time_min", "?"), 1)
        out.append(f"\nHelicopter Route: Helicopter takes {heli_min} minutes to reach the fire.")

    return "\n".join(out)


def get_exact_token_count(model_path: str, text: str) -> int:
    """
    Return the exact token count for a text as seen by the Llama model.
    """
    with suppress_stdout_stderr():
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )
    
    # Tokenize the text
    tokens = llm.tokenize(text.encode('utf-8'))
    return len(tokens)


def query_llm_for_first_map_output(model_path: str, context: str, map_scenario: str, route_info: dict) -> str:
    """Generate structured deployment recommendation for first map output."""
    from llama_cpp import Llama

    print("Running query_llm_for_first_map_output()...")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False
    )

    prompt = f"""
    You are FireGPT, an assistant for wildfire deployments in Germany.

    Your task is to generate a structured deployment recommendation based strictly on the wildfire scenario, fire station data, legal context, and available helicopter resources.

    You must reason clearly, avoid repetition, and always refer to the actual input information. Do not invent facts.
    ---
    ### Output Instructions:
    Write your answer using the **exact headings and structure** below in valid **Markdown**.

    **Requirements:**
    - Deploy exactly **two or three stations** depending on how dangerous the fire is.
    - Only deploy more if it is **really necessary** and there is a exceptional reason that differs from the all the others. 
    - Include at least one deployed fire station WITHOUT **Freiwillige** in the fire station name out of Available Stations info.
    - A **professional fire station** has a name like **“Feuerwehr [City]”** or **“Berufsfeuerwehr [City]”**
    - Choose based on response time, tactical advantage, and resources, due to legal requirements
    - Decide helicopter deployment based on scenario/context
    - Provide response time 
    - Tactical reasoning for deployment (e.g. distance, role, fire size, required and available resources, info from provided context, legal)

    **Reasons to use helicopters and stations:**
    - When justifying the choice of fire station, the reason should always include **the cooperation of the selected units and various tactical and legal reasons**.
    - References to which station is taking on which task are permitted.
    - Each deployed station must have a distinct justification for its involvement and should be assigned a clearly defined, separate operational role, such as coordination, support, or suppression.
    - Write one full paragraph per deployed station. Explain *detailed* why the station is deployed, covering both tactical and legal/contextual reasons. When justifying the choice of fire station, the reason should always include **the cooperation between the selected units**.
    - When assigning roles, ensure that command, suppression, and support tasks are clearly separated among the selected stations.
    - Avoid overlapping responsibilities unless it is explicitly justified.
    - Do not deploy a station if its justification would be the same as another already deployed station.
    - Each deployed unit must have a unique and clearly distinct tactical and/or legal reason for its inclusion. 
    - Do **not** call any station with “Freiwillige” in its name a professional unit.
    - If a professional unit is deployed, always state this directly in the justification.
    - Consider **NOT deploying a helicopter** if:
    - The **fire radius is small**, meaning a fire radius **smaller than 30 meters**.
    - Ground response is fast and sufficient.
    - If the **fire risk is low**.
    - Terrain is easy to access.
    - There is no threat to people or infrastructure.
    - **Always mention** if ther are technical failures or delays prevent its effective use.
    - Whatever your decision, always state clearly why the helicopter is or is not used. 
    - If the helicopter is not deployed, state that explecitly and **suggest how ground units can compensate**, e.g. enhanced coordination, flanking, observation points.
    - Consider deploying a helicopter if:
    - The fire is medium or large, meaning a fire radius greater than or equal to 30 meters, in particular greater than 60 meters.
    - There is threat to people, buildings, or infrastructure.
    - Terrain is difficult for ground units.
    - Ground units have a long ETA or limited access.
    - Aerial support would speed up containment or improve oversight.

    **Reasons not to use in the output:**
    - Never say **'faster arrival time'**, use instead **'fast arrival time'**
    - Do not use same reasons for different fire stations

    **Under each section in Additional Considerations:**
    - Use **complete English sentences**.
    - Use only facts from the input and **justify them as precisely as possible**.
    - Do **not invent** or assume anything.
    - Do **not use bullet points** in the Deployed Stations or Helicopter section.
    - For each station, combine the **tactical and legal/contextual reasoning** into **one clear paragraph**.
    - Try not to **repeat yourself** by giving the same reason twice, unless it is really necessary.
    - In "Additional Considerations", write **2–4 full precisely justified sentences** per item, only more if good reason.
    - Use [Optional:] only if detailed information about this field is provided and has not already been answered by the others (NO redundant information). 
    ---
    ### Markdown Output Format:

    ## Deployed Stations:
    - **[Deployed Fire Station Name] – [ETA in minutes]:** [Write one full paragraph per deployed station. Explain *detailed* why the station is deployed, covering both tactical and legal/contextual reasons. When justifying the choice of fire station, the reason should always include **the cooperation between the selected units**.] 
    - [Next fire station]

    ## Helicopter:
    **[Yes or No] – [ETA in minutes]:** [One or two complete sentences explaining whether helicopter support is used tactically and/or legally and why or why it is not used. Always justify the use or non-use of helicopters on tactical and coordinative grounds.]

    ## Additional Considerations:
    - **Reporting duties:** Describe whether the fire must be reported, to whom, and why.
    - **Tactics:** Explain coordination strategy, use of resources, and response timing.
    - **Terrain/weather:** Describe terrain or weather impacts on fire spread or unit access.
    - **Legal rules:** List applicable legal obligations (e.g., use of professional units, reporting thresholds).
    - **[Optional:] Communication & coordination**
    - **[Optional:] Safety**
    ---    
    ### Example (**For structure only – do NOT USE EXAMPLE informations in the output**)::

    ## Deployed Stations:
    **Professional fire station Regensburg - 15 min:** This non-volunteer unit is deployed in line with legal requirements and assumes full command of the operation. It is responsible for coordination, strategic planning, incident leadership, and inter-agency communication. Thanks to its professional staffing and equipment, it leads complex operations in strategic or densely populated areas with a centralized approach.
    **Freiwillige Feuerwehr Kelheim - 8 min:** Deployed in a supporting role under the command of the professional unit, this volunteer force contributes manpower, water logistics, and perimeter control. As part of Bavaria’s decentralized system, it expands operational capacity without duplicating leadership or decision-making roles.
    ## Helicopter:
    **Yes – 5.1 min:** The helicopter is deployed primarily for its rapid response capability and its versatility in reaching areas that are inaccessible or difficult to access by ground vehicles. It provides a strategic aerial overview, supports reconnaissance, and can assist in coordinating ground units more effectively. Depending on equipment, it may be used for water drops or the transport of personnel and critical supplies

    ## Additional Considerations:
    - **Reporting duties:** Since the affected area exceeds a radius of 100 metres, the fire must be reported to the State Ministry of Food, Agriculture and Forestry.
    - **Tactics:** The combination of professional and volunteer units, supported by aerial response, ensures a fast and balanced deployment. Quick containment is essential due to moderate fire risk.
    - **Terrain/weather:** The fire occurs on non-irrigated arable land with moderate humidity. Accessibility is good, but dryness increases the spread potential.
    - **Legal rules:** Bavarian law requires the deployment of at least one professional fire station. This is fulfilled by Regensburg. Fires threatening disaster must also be reported to the Situation Centre of the Interior Ministry.
    - **Communication & coordination:** Due to the fire's location in a protected forest reserve, coordination with the local forestry office and environmental authority is legally required. This ensures proper handling of sensitive habitats and reporting duties.
    ---
    Now write the answer for the following input:

    ### Wildfire Scenario:
    {map_scenario}

    ### Available Stations and Helicopter Info:
    {_format_route_info(route_info)}

    ### Context:
    {context}

    ---
    Return only the structured answer in Markdown format directly here.
    """
    print(prompt)
    print("Prompt token count:", get_exact_token_count(model_path, prompt))

    print(f"Infos: {_format_route_info(route_info)}")
    
    try:
        set_rag_status(f"---- Generating Final Output ---- ")
        response = llm(
            prompt,
            max_tokens=4000,  # Maximum number of tokens to generate in the response
            stop=["END_OF_RESPONSE"],
            echo=False,
            stream=False,     # Must be False to get a dict
            seed=42,  # seed 
            temperature=0.5 #temperature
        )

        print("LLM response in query code:\n", response)

        if response.get("choices"):
            return response["choices"][0]["text"].strip()
        else:
            return "No meaningful response could be generated."

    except Exception as e:
        print(f"LLM error: {e}")
        return "LLM error – no response generated."


def query_normal_chat_llm(model_path: str,  conversation_history: list,current_question: str,full_context: str, max_history_length: int = 5) -> str:
    """
    Normal chat function with conversation history.
    
    Args:
        model_path: Path to GGUF model
        conversation_history: List of all previous messages
        current_question: Current user question
        full_context: Full context for the question
        max_history_length: Max number of Q&A pairs in context
    
    Returns:
        LLM response as string
    """
    print(f"--- Normal Chat LLM (History: {len(conversation_history)//2} Q&A pairs) ---")
    
    with suppress_stdout_stderr():
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=8192,  # 8K Context
            verbose=False
        )

    # Create chat prompt
    prompt = """
    You are a wildfire assistant supporting a local commander in active wildfire situations in Germany.

    Your role:
    - Always focus only on answering the **current question**
    - Consider any useful context from previous deployments, the conversation history, and the operational environment.
    - If something changes or fails (e.g. helicopter fails to arrive), provide **an adaptive recommendation** based on deployed fire stations, the remaining resources and expected fire behavior.
    - Prioritize **human safety**, **tactical coordination**, and **legal requirements**.
    - Use **clear**, **concise**, and **actionable** instructions.
    - Highlight key actions or concepts using double asterisks, such as **evacuation**, **monitor**, **support**, **alert**, **terrain**, **coordination**, **reporting**, **fallback**, or **reallocate**.
    - Do **not repeat the same noun or unit more than twice** in a single response.
    - Do **not repeat** the same tactical justification or phrase across multiple sentences.

    Response Style:
    - Always respond in **concise, paragraph-style text**.
    - Do **not** explain your internal reasoning.
    - Do **not** answer questions unrelated to wildfire response.
    - Do **not** just repeat the facts listed in the conversation history, insted use them to provide tactically sound, context-specific recommendations that directly address the current question.
    - Do **not** use labels like “Answer:”
    - Make sure the response is grounded in **the known deployment** and **available backup options**.
    - If a resource (like a helicopter) fails, propose what to do **instead** based on the current deployment and scenario.
    - Highlight if used key terms like **evacuation**, **coordination**, **alert**, **monitor**, **terrain**, **reporting**, **support**, **guide** with double asterisks (**like this**).

    ---

    You are now asked to respond to a real situation.  
    Only answer the **current question** directly below.  
    Use previous context (such as unit deployment, fire size, terrain, or weather) to provide further tactical guidance, fallback strategies, or escalation recommendations — especially if current resources become unavailable or conditions change.

    ---

    ### Conversation history:
    """

    print(f"Size after beginning prompt: {get_exact_token_count(model_path, prompt)}")


    total_qa_pairs = len(conversation_history) // 2
    pairs_to_use = min(max_history_length, total_qa_pairs)

    messages_to_use = pairs_to_use * 2
    recent_history = conversation_history[-messages_to_use:]

    prompt += "Conversation history:\n"
    for msg in recent_history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            # For Assistant messages: shorten only if very long
            content = msg['content']
            prompt += f"Assistant: {content}\n"

    # Current question
    prompt += f"\n Current question (**Answer ONLY this question, NONE OTHER**, and use only the necessary historical information and context if it is helpful in answering this question.): {current_question}\n"


    # Add context
    prompt += f"\nContext:\n{full_context}\n"

    prompt += f"\Answer:\n"

    print("\n--- LLM Prompt ---")
    print(prompt)

    print("Prompt token count:", get_exact_token_count(model_path, prompt))

    set_rag_status(f"---- Generating Follow-up Output ---- ")
    # Query LLM
    response = llm(
            prompt,
            max_tokens=4000,  # Maximum number of tokens to generate in the response
            stop=["END_OF_RESPONSE", "**Note:**", "\n\n**" ],
            echo=False,
            stream=False,     # Must be False to get a dict
            seed=42,  # seed 
            temperature=0.7 #temperature
        )
    
    if isinstance(response, dict) and 'choices' in response and response['choices']:
        return response['choices'][0]['text'].strip()
    else:
        print(f"Unexpected LLM response format: {response}")
        return "Sorry, I could not generate a response."