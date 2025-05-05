###CODE:

import json
import os
import sys
import google.generativeai as genai
from google.api_core import exceptions
import re
import time # For potential delays/retries
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
QA_BANK_FILE = "qa_bank.json"
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest" # Or other suitable model
# Maximum context window (tokens) for the model. Verify for your specific model.
MAX_CONTEXT_WINDOW = 32000
# Target proportion of the context window for history (summary + verbatim turns)
HISTORY_TOKEN_BUDGET_RATIO = 0.80 # Increased slightly, more buffer needed for prompt/response
# Number of recent turns to always keep verbatim
LAST_VERBATIM_TURNS = 10
# Estimated max tokens for the non-history part of the prompt (instructions, question, etc.)
# Helps calculate a target for summary re-compression. Adjust as needed.
PROMPT_OVERHEAD_ESTIMATE = 500

# --- Global Model Instance (initialized later) ---
model_instance = None

# --- Helper Functions ---

def load_qa_bank(filename):
    """Loads the question and answer bank from a JSON file. Validates structure."""
    # (Validation code remains the same as in previous versions)
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- Validation ---
        if "concepts" not in data or not isinstance(data["concepts"], list):
            print(f"Error: JSON file '{filename}' must contain a top-level key 'concepts' which is a list.", file=sys.stderr)
            sys.exit(1)
        concepts = data["concepts"]
        if not concepts: print(f"Error: 'concepts' list in '{filename}' is empty.", file=sys.stderr); sys.exit(1)
        for i, concept in enumerate(concepts):
            if not isinstance(concept, dict): print(f"Error: Item {i+1} in 'concepts' list is not a dictionary.", file=sys.stderr); sys.exit(1)
            required_keys = ["concept_name", "stem_misperception", "socratic_questions"]
            for key in required_keys:
                if key not in concept: print(f"Error: Concept {i+1} (index {i}) is missing required key '{key}'.", file=sys.stderr); sys.exit(1)
            if not isinstance(concept["concept_name"], str) or not concept["concept_name"].strip(): print(f"Error: Concept {i+1} (index {i}) has an invalid or empty 'concept_name'.", file=sys.stderr); sys.exit(1)
            if not isinstance(concept["stem_misperception"], str) or not concept["stem_misperception"].strip(): print(f"Error: Concept {i+1} (index {i}) has an invalid or empty 'stem_misperception'.", file=sys.stderr); sys.exit(1)
            if not isinstance(concept["socratic_questions"], list): print(f"Error: Concept {i+1} (index {i})'s 'socratic_questions' is not a list.", file=sys.stderr); sys.exit(1)
            if not concept["socratic_questions"]: print(f"Warning: Concept '{concept['concept_name']}' (index {i}) has an empty 'socratic_questions' list. It will be skipped.", file=sys.stderr)
            for j, question in enumerate(concept["socratic_questions"]):
                 if not isinstance(question, str) or not question.strip(): print(f"Error: Question {j+1} (index {j}) in Concept '{concept['concept_name']}' (index {i})'s 'socratic_questions' is invalid or empty.", file=sys.stderr); sys.exit(1)
        print(f"Successfully loaded {len(concepts)} concepts from '{filename}'.")
        return concepts
    except FileNotFoundError: print(f"Error: QA Bank file '{filename}' not found.", file=sys.stderr); sys.exit(1)
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from file '{filename}'. Check its format.", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"An unexpected error occurred while loading '{filename}': {e}", file=sys.stderr); sys.exit(1)


def configure_gemini():
    """Configures the Gemini API and returns the model instance."""
    global model_instance
    if not API_KEY: print("Error: GOOGLE_API_KEY not found.", file=sys.stderr); sys.exit(1)
    try:
        genai.configure(api_key=API_KEY)
        model_instance = genai.GenerativeModel(MODEL_NAME)
        print(f"Successfully configured Gemini model: {MODEL_NAME}")
        # Verify token counting
        model_instance.count_tokens("test")
        print("Token counting capability verified.")
        return model_instance
    except exceptions.PermissionDenied as e: print(f"Error: Permission denied configuring Gemini API. Check API key/permissions. Details: {e}", file=sys.stderr); sys.exit(1)
    except exceptions.GoogleAPIError as e: print(f"Error configuring Gemini API or loading model: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"An unexpected error occurred during Gemini configuration: {e}", file=sys.stderr); sys.exit(1)


def get_student_discipline():
    """Asks the student to select their primary STEM discipline."""
    # (Function remains the same as in previous versions)
    disciplines = ["Mathematics", "Engineering", "Physics", "Other STEM"]
    print("\nSelect your primary STEM discipline:")
    for i, discipline in enumerate(disciplines): print(f"{i + 1}. {discipline}")
    while True:
        try:
            choice = input(f"Discipline ({'/'.join([d for d in disciplines])} number, or 'quit'): ").strip().lower()
            if choice == 'quit': sys.exit(print("Exiting tutor session. Goodbye!"))
            if not choice.isdigit(): print("Invalid input."); continue
            idx = int(choice) - 1
            if 0 <= idx < len(disciplines): print(f"Selected: {disciplines[idx]}"); return disciplines[idx]
            else: print(f"Invalid number (1-{len(disciplines)}).")
        except (EOFError, KeyboardInterrupt): sys.exit(print("\nExiting tutor session. Goodbye!"))
        except ValueError: print("Invalid input.")


def count_tokens(text):
    """Counts tokens using the configured Gemini model. Handles errors."""
    global model_instance
    if not model_instance: print("Error: Model not configured for token counting.", file=sys.stderr); return 0
    if not text: return 0 # Empty string has 0 tokens
    try:
        response = model_instance.count_tokens(text)
        return response.total_tokens
    except exceptions.GoogleAPIError as e: print(f"\nWarning: API error during token counting: {e}. Returning 0.", file=sys.stderr); return 0
    except Exception as e: print(f"\nWarning: Unexpected error during token counting: {e}. Returning 0.", file=sys.stderr); return 0

def format_history_for_prompt(summary, verbatim_turns):
    """Formats the summary and verbatim turns into a string for the prompt context."""
    history_parts = []
    if summary:
        history_parts.append("Summary of Older Conversation Turns:")
        summary_lines = summary.split('\n')
        indented_summary = "\n  ".join(summary_lines)
        history_parts.append(f"  {indented_summary}")
        history_parts.append("\nRecent Verbatim Conversation Turns:")
    else:
        history_parts.append("Conversation History (Recent Turns Only):")

    if not verbatim_turns:
        if not summary:
            return "No conversation history for this concept yet."
        else:
             # Should not happen if LAST_VERBATIM_TURNS > 0 and history exists
             history_parts.append("  (No recent verbatim turns available)")
    else:
        for turn in verbatim_turns:
            role = turn.get('role', 'unknown').capitalize()
            content = turn.get('content', '').strip()
            content_lines = content.split('\n')
            indented_content = "\n  ".join(content_lines)
            history_parts.append(f"- {role}:\n  {indented_content}")

    return "\n".join(history_parts)

def summarize_history(model, turns_to_summarize, discipline, target_tokens=None):
    """
    Calls the AI to summarize the provided history turns.
    Optionally accepts a target_tokens hint to guide summary length.
    """
    if not turns_to_summarize:
        return "" # No turns to summarize

    print(f"\nSummarizing {len(turns_to_summarize)} older conversation turn(s)...")

    turns_text_list = [f"{t.get('role', '??').capitalize()}: {t.get('content', '').strip()}" for t in turns_to_summarize]
    turns_text = "\n".join(turns_text_list)

    # Construct the prompt with optional length guidance
    length_guidance = ""
    if target_tokens and target_tokens > 0:
        # Provide a word count estimate as a proxy for tokens, as models handle word counts better
        # Adjust the words/token ratio based on observation if needed (e.g., 3 words/token)
        estimated_words = int(target_tokens * 0.7) # Rough estimate
        length_guidance = f"Aim for a concise summary, ideally around {estimated_words} words (approx. {target_tokens} tokens)."
        print(f"(Attempting to limit summary to ~{target_tokens} tokens)")

    prompt = f"""
    Context:
    You are an AI assistant condensing a conversation history for an ongoing tutoring session with a {discipline} student.
    Create a concise summary of the following turns, retaining essential information: key concepts, student misunderstandings, tutor clarifications, and significant examples.

    Conversation Turns to Summarize:
    ---
    {turns_text}
    ---

    Task:
    Generate a concise summary focusing on:
    - Main economic topics covered.
    - Recurring student misconceptions or difficulties, especially those contrasting with {discipline} intuition.
    - Key clarifications, insights, or corrections provided by the tutor.
    - Important examples/analogies used.
    - Maintain a neutral, objective tone.
    {length_guidance}

    Concise Summary:
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3)
        )

        if not response._result.candidates or not response.parts or not hasattr(response.parts[0], 'text'):
             print("\nWarning: AI summarization failed. Summary may be incomplete.", file=sys.stderr)
             return "[Summarization failed]"

        new_summary = response.text.strip()
        summary_tokens = count_tokens(new_summary)
        print(f"Summarization complete ({summary_tokens} tokens).")
        return new_summary

    except exceptions.GoogleAPIError as e:
        print(f"\nWarning: API error during summarization: {e}. Summary may be incomplete.", file=sys.stderr)
        return "[Summarization failed due to API error]"
    except Exception as e:
        print(f"\nWarning: Unexpected error during summarization: {e}. Summary may be incomplete.", file=sys.stderr)
        return "[Summarization failed due to unexpected error]"


def manage_history_and_get_context(model, current_summary, chat_history, discipline):
    """
    Manages history by summarizing old turns and ensuring the context fits the budget.
    Prioritizes keeping the last LAST_VERBATIM_TURNS verbatim.
    Returns the formatted context string and potentially updated summary.
    """
    history_token_budget = int(MAX_CONTEXT_WINDOW * HISTORY_TOKEN_BUDGET_RATIO)

    # 1. Split history into verbatim and older sections
    verbatim_cutoff = max(0, len(chat_history) - LAST_VERBATIM_TURNS)
    verbatim_turns = chat_history[verbatim_cutoff:]
    older_turns = chat_history[:verbatim_cutoff]

    # 2. Check if summarization/re-summarization of older turns is needed
    # We re-summarize *all* older turns if the older_turns list is non-empty.
    # This simplifies logic compared to tracking *new* older turns.
    if older_turns:
        # Summarize all turns designated as 'older'
        current_summary = summarize_history(model, older_turns, discipline)
    else:
        # If there are no older turns, ensure summary is empty
        current_summary = ""

    # 3. Calculate token counts for current summary and verbatim turns
    summary_tokens = count_tokens(current_summary)
    verbatim_turns_text = "\n".join([f"{t['role']}: {t['content']}" for t in verbatim_turns])
    verbatim_tokens = count_tokens(verbatim_turns_text)

    total_tokens = summary_tokens + verbatim_tokens

    # 4. Handle overflow: Re-summarize summary if needed to fit budget
    if total_tokens > history_token_budget:
        print(f"\nWarning: Combined history ({total_tokens} tokens) exceeds budget ({history_token_budget}). Re-summarizing older turns...")

        # Calculate how many tokens are allowed for the summary
        allowed_summary_tokens = history_token_budget - verbatim_tokens

        if allowed_summary_tokens < 0:
            # Edge case: Verbatim turns alone exceed budget. Cannot proceed with history.
            print(f"ERROR: Verbatim turns ({verbatim_tokens} tokens) alone exceed budget ({history_token_budget}). Discarding history.", file=sys.stderr)
            # Optionally, could try truncating verbatim turns, but for now, discard all.
            current_summary = "[History discarded due to excessive verbatim length]"
            verbatim_turns = [] # Clear verbatim turns as well
        elif allowed_summary_tokens < summary_tokens and older_turns:
            # Summary is too long, and there *are* older turns to summarize
            print(f"Targeting summary size: ~{allowed_summary_tokens} tokens.")
            # Re-summarize older turns with a target token count
            current_summary = summarize_history(model, older_turns, discipline, target_tokens=allowed_summary_tokens)
            # Re-check total tokens after re-summarization (it's an estimate)
            summary_tokens = count_tokens(current_summary)
            total_tokens = summary_tokens + verbatim_tokens
            if total_tokens > history_token_budget:
                 print(f"Warning: Re-summarization still resulted in {total_tokens} tokens (budget {history_token_budget}). Summary might be truncated by model.")
        elif not older_turns:
             # Overflow caused only by verbatim turns, but they fit within budget alone. No summary exists/needed.
             current_summary = ""


    # 5. Format the final context string using the final summary and verbatim turns
    history_context_string = format_history_for_prompt(current_summary, verbatim_turns)

    # Return the context string and the potentially updated summary
    # chat_history list itself is not modified here, only used as input
    return history_context_string, current_summary


# --- AI Interaction Functions (Modified to accept formatted history string) ---
# These functions (get_ai_feedback, get_ai_hint, get_ai_scaffolded_explanation)
# remain identical to the previous version where they accept 'history_context_string'.
# No changes needed within them for this new strategy.

def get_ai_feedback(model, concept, question, user_answer, discipline, history_context_string):
    """Gets feedback, using the pre-formatted history context string."""
    # --- Discipline Guidance (Same as before) ---
    if discipline == "Mathematics": discipline_guidance = "..." # Keep full text
    elif discipline == "Engineering": discipline_guidance = "..." # Keep full text
    elif discipline == "Physics": discipline_guidance = "..." # Keep full text
    else: discipline_guidance = "..." # Keep full text
    # --- End Discipline Guidance ---

    prompt = f"""
    Context:
    You are an AI Economics Tutor interacting with a {discipline} student. Use the conversation history for context.

    {history_context_string}

    Concept: "{concept.get('concept_name', 'N/A')}" (Potential STEM Misconception: "{concept.get('stem_misperception', 'N/A')}")

    Task:
    Analyze the student's response below. Provide encouraging feedback (2-6 sentences) tailored to {discipline}.
    1. CRITICAL: Analyze ALL parts of the student's answer methodically. Address every substantive point.
    2. Validate ONLY explicitly correct points using their wording. If none, just acknowledge.
    3. Gently identify gaps/misunderstandings, contrasting with {discipline} intuition if applicable.
    4. Provide at least one CONCRETE example with numbers/quantities.
    5. Draw explicit connections/contrasts to {discipline} concepts/terminology.
    6. Include a relevant mathematical/visual representation (equation for Math students).
    7. IMPORTANT: Use diverse economic principles, avoid overusing "subjective value". Emphasize different principles per question.
    {discipline_guidance}

    Constraint: Do NOT ask follow-up questions. Focus on feedback for the given answer. Do NOT infer understanding.

    Socratic Question Asked: "{question}"
    Student's Answer: "{user_answer}"

    Your Feedback (as Economics Tutor, tailored for {discipline} student):
    """
    # (Rest of the function: API call, error handling - same as before)
    try:
        safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, safety_settings=safety_settings, generation_config=genai.types.GenerationConfig(temperature=0.4))
        if not response._result.candidates: reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"; return f"[AI Feedback Blocked: {reason}] Please rephrase."
        if not response.parts or not hasattr(response.parts[0], 'text'): return "[AI Feedback Unavailable: No text content]"
        return response.text.strip()
    except exceptions.GoogleAPIError as e: print(f"\nError during feedback: {e}", file=sys.stderr); return "[AI Feedback Error: API error.]"
    except Exception as e: print(f"\nUnexpected error during feedback: {e}", file=sys.stderr); return "[AI Feedback Error: Unexpected issue.]"

def get_ai_hint(model, concept, question, discipline, history_context_string):
    """Gets a hint, using the pre-formatted history context string."""
    # --- Discipline Guidance (Same as before) ---
    if discipline == "Mathematics": discipline_guidance = "..." # Keep full text
    elif discipline == "Engineering": discipline_guidance = "..." # Keep full text
    elif discipline == "Physics": discipline_guidance = "..." # Keep full text
    else: discipline_guidance = "..." # Keep full text
    # --- End Discipline Guidance ---

    prompt = f"""
    Context:
    AI Economics Tutor providing a hint to a {discipline} student stuck on a question (no answer given yet). Use conversation history.

    {history_context_string}

    Concept: "{concept.get('concept_name', 'N/A')}" (Potential STEM Misconception: "{concept.get('stem_misperception', 'N/A')}")

    Task:
    Provide a brief (2-4 sentences) hint for the question below.
    - CRITICAL: Start with neutral openers ("Consider...", "Think about..."), NEVER validating language.
    - Nudge towards the correct economic perspective.
    - Suggest a related concept/framework from {discipline} and contrast it with economics.
    - Provide a CONCRETE EXAMPLE with specific numbers/values.
    - Avoid the direct answer. Be encouraging without false validation.
    {discipline_guidance}

    Socratic Question: "{question}"

    Your Hint (as Economics Tutor, tailored for {discipline} student):
    """
    # (Rest of the function: API call, error handling - same as before)
    try:
        safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, safety_settings=safety_settings, generation_config=genai.types.GenerationConfig(temperature=0.6))
        if not response._result.candidates: reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"; return f"[AI Hint Blocked: {reason}] Could not generate hint."
        if not response.parts or not hasattr(response.parts[0], 'text'): return "[AI Hint Unavailable: No text content]"
        return response.text.strip()
    except exceptions.GoogleAPIError as e: print(f"\nError during hint: {e}", file=sys.stderr); return "[AI Hint Error: API error.]"
    except Exception as e: print(f"\nUnexpected error during hint: {e}", file=sys.stderr); return "[AI Hint Error: Unexpected issue.]"


def get_ai_scaffolded_explanation(model, concept, question, discipline, history_context_string):
    """Provides a scaffolded explanation, using the pre-formatted history context string."""
    # --- Discipline Guidance (Same as before) ---
    if discipline == "Mathematics": discipline_guidance = "..." # Keep full text
    elif discipline == "Engineering": discipline_guidance = "..." # Keep full text
    elif discipline == "Physics": discipline_guidance = "..." # Keep full text
    else: discipline_guidance = "..." # Keep full text
    # --- End Discipline Guidance ---

    prompt = f"""
    Context:
    AI Economics Tutor helping a stuck/confused {discipline} student. Use conversation history.

    {history_context_string}

    Concept: "{concept.get('concept_name', 'N/A')}" (Potential STEM Misconception: "{concept.get('stem_misperception', 'N/A')}")

    Task:
    Provide a scaffolded explanation (5-8 sentences) for the question below.
    1. Acknowledge difficulty encouragingly.
    2. Use THREE TIERS: Basic definition ({discipline} terms), CONCRETE example (numbers), Connect to the question.
    3. Draw parallel to {discipline} concept (similarity & difference, using {discipline} terms).
    4. Suggest a first step leveraging their background.
    5. Avoid the direct answer. Use diverse economic principles.
    {discipline_guidance}

    Socratic Question stuck on: "{question}"

    Your Scaffolded Explanation (as Economics Tutor, tailored for {discipline} student):
    """
    # (Rest of the function: API call, error handling - same as before)
    # NOTE: Ellipses (...) above in discipline_guidance indicate the full text from previous versions should be inserted there.
    try:
        safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, safety_settings=safety_settings, generation_config=genai.types.GenerationConfig(temperature=0.7))
        if not response._result.candidates: reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"; return f"[AI Explanation Blocked: {reason}] Could not generate explanation."
        if not response.parts or not hasattr(response.parts[0], 'text'): return "[AI Explanation Unavailable: No text content]"
        return response.text.strip()
    except exceptions.GoogleAPIError as e: print(f"\nError during explanation: {e}", file=sys.stderr); return "[AI Explanation Error: API error.]"
    except Exception as e: print(f"\nUnexpected error during explanation: {e}", file=sys.stderr); return "[AI Explanation Error: Unexpected issue.]"


# --- Main Tutor Logic ---

def run_tutor():
    """Runs the main Socratic tutoring session with the new history strategy."""
    concepts = load_qa_bank(QA_BANK_FILE)
    model = configure_gemini()
    discipline = get_student_discipline()

    print("\n--- Socratic Economics Tutor for STEM ---")
    print(f"Discipline: {discipline}. History Strategy: Last {LAST_VERBATIM_TURNS} verbatim, older summarized.")
    print(f"Managing history within ~{int(MAX_CONTEXT_WINDOW * HISTORY_TOKEN_BUDGET_RATIO)} token budget.")
    print("'quit', 'menu', 'hint'")
    print("-----------------------------------------\n")

    i_dont_know_pattern = re.compile(
        r"^\s*(i\s+don'?t\s+know|no\s+idea|not\s+sure|confused|clueless|stuck|help|guide\s+me)\s*$",
        re.IGNORECASE
    )

    while True: # Concept selection loop
        print("\n" + "="*60 + "\nChoose a concept:")
        valid_concepts = [c for c in concepts if c.get("socratic_questions") and c.get("concept_name")]
        if not valid_concepts: print("No concepts available. Exiting."); sys.exit(0)
        for i, concept in enumerate(valid_concepts): print(f"{i + 1}. {concept.get('concept_name', f'Concept {i+1}')}")
        print("\nType number or 'quit'.")

        while True: # Input loop for concept choice
            try:
                choice = input("Choice: ").strip().lower()
                if choice == 'quit': sys.exit(print("Exiting. Goodbye!"))
                if not choice.isdigit(): print("Invalid input."); continue
                idx = int(choice) - 1
                if 0 <= idx < len(valid_concepts): selected_concept = valid_concepts[idx]; break
                else: print(f"Invalid number (1-{len(valid_concepts)}).")
            except (EOFError, KeyboardInterrupt): sys.exit(print("\nExiting. Goodbye!"))
            except ValueError: print("Invalid input.")

        # --- Run session for the selected concept ---
        chat_history = [] # Stores ALL turns: {'role': ..., 'content': ...}
        current_summary = "" # Running summary of turns older than LAST_VERBATIM_TURNS
        return_to_menu = False

        print("\n" + "="*60)
        print(f"Concept: {selected_concept.get('concept_name', 'N/A')}")
        print(f"STEM Misconception Focus: {selected_concept.get('stem_misperception', 'N/A')}")
        print("="*60 + "\n")

        questions = selected_concept.get("socratic_questions", [])
        total_questions = len(questions)

        for j, question_text in enumerate(questions):
            if not questions: break

            question_num = j + 1
            print(f"\n-- Question {question_num}/{total_questions} --")
            print(f"Q: {question_text}")

            while True: # Inner loop for user answer/command
                 try:
                     # *** Manage history BEFORE getting input ***
                     # This updates summary based on older turns and handles budget limits
                     history_context_string, current_summary = manage_history_and_get_context(
                         model, current_summary, chat_history, discipline
                     )
                     # Note: chat_history list itself isn't modified by the function above

                     user_input = input("Your Answer ('hint', 'menu', 'quit', or answer): ").strip()
                     lower_input = user_input.lower()

                     if lower_input == 'quit': sys.exit(print("\nExiting. Goodbye!"))
                     if lower_input == 'menu': print("\nReturning to menu..."); return_to_menu = True; break # Exit inner loop
                     if lower_input == 'hint':
                          print("Generating hint...")
                          hint = get_ai_hint(model, selected_concept, question_text, discipline, history_context_string)
                          print(f"\nAI Tutor Hint:\n{hint}\n")
                          # Append hint request and response to the FULL history
                          chat_history.append({"role": "user", "content": "(Requested a hint)"})
                          chat_history.append({"role": "assistant", "content": hint})
                          # Continue prompt loop (history will be re-managed on next iteration)
                          continue
                     if i_dont_know_pattern.match(lower_input):
                          print("Okay, let's break that down...")
                          explanation = get_ai_scaffolded_explanation(model, selected_concept, question_text, discipline, history_context_string)
                          print(f"\nAI Tutor Explanation:\n{explanation}\n")
                          # Append confusion and response to FULL history
                          chat_history.append({"role": "user", "content": user_input})
                          chat_history.append({"role": "assistant", "content": explanation})
                          # Continue prompt loop
                          continue
                     if not user_input: print("Please enter an answer or use a command."); continue

                     # --- Process valid user answer ---
                     user_answer = user_input
                     # Append answer to FULL history FIRST
                     chat_history.append({"role": "user", "content": user_answer})

                     # *** Manage history AGAIN before getting feedback (includes new user answer) ***
                     history_context_string, current_summary = manage_history_and_get_context(
                         model, current_summary, chat_history, discipline
                     )

                     print("Analyzing answer...")
                     feedback = get_ai_feedback(model, selected_concept, question_text, user_answer, discipline, history_context_string)
                     print(f"\nAI Tutor Feedback:\n{feedback}\n")
                     # Append feedback to FULL history
                     chat_history.append({"role": "assistant", "content": feedback})

                     print("-" * 60)
                     break # Exit inner loop, proceed to next question

                 except (EOFError, KeyboardInterrupt): sys.exit(print("\nExiting. Goodbye!"))

            if return_to_menu: break # Exit question loop for this concept

        # --- End of questions ---
        if not return_to_menu:
             print("\n" + "="*60 + f"\nCompleted Concept: {selected_concept.get('concept_name', 'N/A')}\n" + "="*60 + "\n")

# --- Script Entry Point ---
if __name__ == "__main__":
    # Make sure to re-insert the full discipline_guidance text in the get_ai_... functions where marked "..."
    run_tutor()
