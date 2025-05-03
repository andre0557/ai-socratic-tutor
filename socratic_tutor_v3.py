import json
import os
import sys
import google.generativeai as genai
from google.api_core import exceptions
import re
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
QA_BANK_FILE = "qa_bank.json"
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest" # Or potentially gemini-1.5-pro-latest for better reasoning if needed and budget allows

# --- Helper Functions ---

def load_qa_bank(filename):
    """
    Loads the question and answer bank from a JSON file.
    Validates the basic structure.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "concepts" not in data or not isinstance(data["concepts"], list):
            print(f"Error: JSON file '{filename}' must contain a top-level key 'concepts' which is a list.", file=sys.stderr)
            sys.exit(1)

        concepts = data["concepts"]
        if not concepts:
             print(f"Error: 'concepts' list in '{filename}' is empty.", file=sys.stderr)
             sys.exit(1)

        for i, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                 print(f"Error: Item {i+1} in 'concepts' list is not a dictionary.", file=sys.stderr)
                 sys.exit(1)
            required_keys = ["concept_name", "stem_misperception", "socratic_questions"]
            for key in required_keys:
                if key not in concept:
                    print(f"Error: Concept {i+1} (index {i}) is missing required key '{key}'.", file=sys.stderr)
                    sys.exit(1)
            if not isinstance(concept["concept_name"], str) or not concept["concept_name"].strip():
                 print(f"Error: Concept {i+1} (index {i}) has an invalid or empty 'concept_name'.", file=sys.stderr)
                 sys.exit(1)
            if not isinstance(concept["stem_misperception"], str) or not concept["stem_misperception"].strip():
                 print(f"Error: Concept {i+1} (index {i}) has an invalid or empty 'stem_misperception'.", file=sys.stderr)
                 sys.exit(1)
            if not isinstance(concept["socratic_questions"], list):
                 print(f"Error: Concept {i+1} (index {i})'s 'socratic_questions' is not a list.", file=sys.stderr)
                 sys.exit(1)
            if not concept["socratic_questions"]:
                 print(f"Warning: Concept '{concept['concept_name']}' (index {i}) has an empty 'socratic_questions' list. It will be skipped.", file=sys.stderr)
                 # Keep going, just warn
            for j, question in enumerate(concept["socratic_questions"]):
                 if not isinstance(question, str) or not question.strip():
                      print(f"Error: Question {j+1} (index {j}) in Concept '{concept['concept_name']}' (index {i})'s 'socratic_questions' is invalid or empty.", file=sys.stderr)
                      sys.exit(1)
        print(f"Successfully loaded {len(concepts)} concepts from '{filename}'.")
        return concepts
    except FileNotFoundError:
        print(f"Error: QA Bank file '{filename}' not found.", file=sys.stderr)
        print("Please ensure the JSON file exists and is in the same directory.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{filename}'. Check its format.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading '{filename}': {e}", file=sys.stderr)
        sys.exit(1)


def configure_gemini():
    """
    Configures the Gemini API and returns the model instance.
    Exits if API Key is not found or configuration fails.
    """
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found.", file=sys.stderr)
        print("Please ensure it is set in your .env file or environment variables.", file=sys.stderr)
        sys.exit(1)

    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Successfully configured Gemini model: {MODEL_NAME}")
        return model
    except Exception as e:
        print(f"Error configuring Gemini API or loading model: {e}", file=sys.stderr)
        if "blocked" in str(e).lower() or "access denied" in str(e).lower():
             print("Reason: API access might be blocked or restricted. Check your key and Google Cloud project settings.", file=sys.stderr)
        sys.exit(1)


def get_ai_feedback(model, concept, question, user_answer):
    """
    Gets feedback from the Gemini model on the user's answer.
    Refined prompt to prevent false validation and improve clarity/STEM relevance.
    """
    prompt = f"""
    Context:
    You are an AI Economics Tutor interacting with an undergraduate STEM student (Math, Engineering, or Physics).
    The student may have misconceptions based on over-reliance on principles learned in their STEM fields, which don't always directly apply to economics.
    Your goal is to help them bridge the gap and understand the specific economic reasoning.

    The current economic concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Analyze the student's response below to the Socratic question provided.
    Provide feedback (aim for 2-6 sentences) that is encouraging and educational. Your feedback should:
    1.  **CRITICAL: ONLY validate points that are explicitly correct or demonstrate accurate understanding based on the student's actual words.** If the answer contains **explicitly correct points or demonstrates accurate understanding of relevant concepts**, start by validating ONLY those points using clear phrases like "That's a correct point about..." or "You've accurately identified...". If the answer does not contain clear correct points (e.g., it's minimal, vague, or wrong), **do NOT falsely attribute understanding.** Simply acknowledge the answer received before proceeding.
    2.  Gently identify any conceptual gaps, misunderstandings, or points where STEM intuition might be misleading.
    3.  Briefly clarify economic principles involved, defining any jargon simply and in context.
    4.  Wherever relevant and helpful for a STEM student, draw explicit connections or contrasts to concepts, models, or intuition from Physics, Math, or Engineering (e.g., systems of equations, equilibrium, feedback loops, conservation laws - clearly stating similarities AND differences).
    5.  If applicable, suggest how the economic concept could be represented using mathematical notation or a visual framework relevant to STEM students (e.g., "think of this as intersecting curves on a graph", "it's like solving a system of equations", "consider variables and parameters", "visualize flow in a network"). Do NOT generate equations or diagrams, just describe the framework.
    6.  Build upon any correct parts of their understanding as a foundation *if* point 1 allowed validation.

    Constraint:
    - Do NOT ask follow-up questions. Focus solely on providing commentary/feedback on the *given* answer. Prioritize clarity and relevance to a STEM background. **Do NOT infer understanding.**

    Socratic Question Asked:
    "{question}"

    Student's Answer:
    "{user_answer}"

    Your Feedback (as Economics Tutor):
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(temperature=0.4) # Slightly lower temp to encourage more precise validation
        )

        if not response._result.candidates:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return f"[AI Feedback Blocked: {response.prompt_feedback.block_reason.name}] Please try rephrasing your answer."
             else:
                 return "[AI Feedback Unavailable: No response part received, reason unknown]"

        if not response.parts or not hasattr(response.parts[0], 'text'):
             return "[AI Feedback Unavailable: Response received but no text content found]"

        return response.text.strip()

    except exceptions.GoogleAPIError as e:
        print(f"\nError communicating with Gemini API during feedback: {e}", file=sys.stderr)
        if isinstance(e, exceptions.AuthenticationError):
             return "[AI Feedback Error: Authentication failed. Check your API key.]"
        elif isinstance(e, exceptions.ResourceExhausted):
             return "[AI Feedback Error: API quota exceeded or rate limited. Please try again later.]"
        else:
             return "[AI Feedback Error: An API error occurred.]"
    except Exception as e:
        print(f"\nAn unexpected error occurred during AI feedback generation: {e}", file=sys.stderr)
        return "[AI Feedback Error: An unexpected issue occurred.]"


def get_ai_hint(model, concept, question):
    """
    Gets a hint from the Gemini model for a given concept and question.
    The hint should guide the student without giving the answer away, incorporating concrete ideas and STEM links.
    Critically, it avoids falsely validating when no answer has been given.
    """
    prompt = f"""
    Context:
    You are an AI Economics Tutor providing a hint to an undergraduate STEM student (Math, Engineering, or Physics) who is stuck on a Socratic question. **The student has not yet provided an answer to the question.**
    The concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Provide a brief (1-4 sentences) hint for the Socratic question below.
    The hint should:
    - **CRITICAL: Never begin with validating language or phrases that imply the student has already made progress or provided a correct starting point.**
    - **Start the hint with neutral, encouraging language or a direct suggestion on how to think about the problem.** Use openers such as "Think about...", "Consider...", "A helpful way to approach this is...", "Here's something to consider...", "To get started, think about...".
    - Gently nudge the student towards the correct economic perspective related to the question.
    - Suggest thinking about a related concept from Physics, Math, or Engineering and how it might be similar or different.
    - Offer a very simple, concrete example or comparison related to the question's core idea, if possible.
    - Avoid giving the direct answer.
    - Be encouraging.

    Socratic Question:
    "{question}"

    Your Hint (as Economics Tutor):
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(temperature=0.6)
        )

        if not response._result.candidates:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return f"[AI Hint Blocked: {response.prompt_feedback.block_reason.name}] Could not generate a hint."
             else:
                 return "[AI Hint Unavailable: No response part received, reason unknown]"

        if not response.parts or not hasattr(response.parts[0], 'text'):
             return "[AI Hint Unavailable: Response received but no text content found]"

        return response.text.strip()

    except exceptions.GoogleAPIError as e:
        print(f"\nError communicating with Gemini API during hint generation: {e}", file=sys.stderr)
        if isinstance(e, exceptions.AuthenticationError):
             return "[AI Hint Error: Authentication failed.]"
        elif isinstance(e, exceptions.ResourceExhausted):
             return "[AI Hint Error: API quota exceeded or rate limited.]"
        else:
             return "[AI Hint Error: An API error occurred.]"
    except Exception as e:
        print(f"\nAn unexpected error occurred during AI hint generation: {e}", file=sys.stderr)
        return "[AI Hint Error: An unexpected issue occurred.]"

def get_ai_scaffolded_explanation(model, concept, question):
    """
    Provides a more structured explanation when the student indicates they are stuck or confused.
    Breaks down the concept related to the question, building from foundational principles with STEM relevance.
    """
    prompt = f"""
    Context:
    You are an AI Economics Tutor helping an undergraduate STEM student (Math, Engineering, or Physics) who is completely stuck or confused by a Socratic question.
    The student indicated they don't know the answer or are too confused to respond. Your goal is to provide foundational understanding to help them try again, without giving the answer away.

    The concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Provide a scaffolded explanation (aim for 3-6 sentences) to help the student approach the specific Socratic question below, starting from foundational principles. Your explanation should:
    - Acknowledge their difficulty in an encouraging way.
    - Break down the absolute core economic principle or definition needed to even start thinking about the question, explaining it simply.
    - Provide a basic, concrete example or analogy to illustrate this core principle, ideally one that relates to a STEM concept they might know, clearly explaining the parallel or contrast.
    - Suggest a simple perspective or the *first step* in thinking about the problem.
    - Avoid giving the direct answer to the original question.
    - Encourage them to try answering the question again after this explanation.
    - Maintain trust by responding to their actual level of understanding, without assuming knowledge they haven't demonstrated.

    Socratic Question the student is stuck on:
    "{question}"

    Your Scaffolded Explanation (as Economics Tutor):
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(temperature=0.7)
        )

        if not response._result.candidates:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return f"[AI Explanation Blocked: {response.prompt_feedback.block_reason.name}] Could not generate an explanation."
             else:
                 return "[AI Explanation Unavailable: No response part received, reason unknown]"

        if not response.parts or not hasattr(response.parts[0], 'text'):
             return "[AI Explanation Unavailable: Response received but no text content found]"

        return response.text.strip()

    except exceptions.GoogleAPIError as e:
        print(f"\nError communicating with Gemini API during explanation generation: {e}", file=sys.stderr)
        if isinstance(e, exceptions.AuthenticationError):
             return "[AI Explanation Error: Authentication failed.]"
        elif isinstance(e, exceptions.ResourceExhausted):
             return "[AI Explanation Error: API quota exceeded or rate limited.]"
        else:
             return "[AI Explanation Error: An API error occurred.]"
    except Exception as e:
        print(f"\nAn unexpected error occurred during AI explanation generation: {e}", file=sys.stderr)
        return "[AI Explanation Error: An unexpected issue occurred.]"

# --- Main Tutor Logic ---

def run_tutor():
    """Runs the main Socratic tutoring session with concept selection, hints, and scaffolding for confusion."""
    concepts = load_qa_bank(QA_BANK_FILE)
    model = configure_gemini()

    print("\n--- Welcome to the Socratic Economics Tutor for STEM Students ---")
    print("Bridging the gap between STEM intuition and economic reasoning.")
    print("Type 'quit' at any prompt to exit. Type 'menu' during questions to return to concept selection.")
    print("Type 'hint' during a question for a clue if you're stuck.")
    print("------------------------------------------------------------------\n")

    # Regex to detect common "I don't know" or confusion phrases
    i_dont_know_pattern = re.compile(
        r"^\s*(i\s+don'?t\s+know|no\s+idea|not\s+sure|confused|this\s+is\s+too\s+confusing|can\s+you\s+guide\s+me|help\s+me|stuck|i'?m\s+confused|i\s+do\s+not\s+know)\s*$",
        re.IGNORECASE
    )

    while True: # Main loop for concept selection
        print("\n" + "="*60)
        print("Choose a concept to explore:")
        print("="*60)
        valid_concepts = [c for c in concepts if c.get("socratic_questions") and c.get("concept_name")] # Only list concepts with questions and names
        if not valid_concepts:
             print("No concepts available with questions. Exiting.")
             sys.exit(0)

        for i, concept in enumerate(valid_concepts):
            print(f"{i + 1}. {concept.get('concept_name', f'Concept {i+1}')}")

        print("\nType the number of the concept, or 'quit' to exit.")

        while True: # Input loop for concept selection
            try:
                choice = input("Your choice: ").strip().lower()

                if choice == 'quit':
                    print("Exiting tutor session. Goodbye!")
                    sys.exit(0)

                if not choice.isdigit():
                    print("Invalid input. Please enter a number or 'quit'.")
                    continue

                concept_index_in_list = int(choice) - 1 # Convert to 0-based index for the valid_concepts list

                if 0 <= concept_index_in_list < len(valid_concepts):
                    selected_concept = valid_concepts[concept_index_in_list]
                    break # Valid selection, exit inner loop
                else:
                    print(f"Invalid number. Please choose between 1 and {len(valid_concepts)}.")

            except (EOFError, KeyboardInterrupt):
                print("\nExiting tutor session. Goodbye!")
                sys.exit(0)
            except ValueError:
                 print("Invalid input. Please enter a number or 'quit'.")


        # --- Run session for the selected concept ---
        print("\n" + "="*60)
        print(f"Starting Concept: {selected_concept.get('concept_name', 'Selected Concept')}")
        print(f"Potential STEM Misconception Focus: {selected_concept.get('stem_misperception', 'N/A')}")
        print("="*60 + "\n")
        print("Let's explore this concept through Socratic questions.")

        questions = selected_concept.get("socratic_questions", [])
        total_questions = len(questions)

        # Flag to check if we should return to menu after question loop
        return_to_menu = False

        for j, question_text in enumerate(questions):
            # Add check here in case questions list was empty despite validation warning
            if not questions: break

            question_num = j + 1
            print(f"\n-- Question {question_num}/{total_questions} --")
            print(f"Q: {question_text}")

            # --- Inner loop for getting user answer, hint, menu, or confusion ---
            while True:
                 try:
                     user_input = input("Your Answer ('hint', 'menu', 'quit', or answer): ").strip()

                     lower_input = user_input.lower() # Use lower for command and keyword checks

                     if lower_input == 'quit':
                          print("\nExiting tutor session. Goodbye!")
                          sys.exit(0)
                     elif lower_input == 'menu':
                          print("\nReturning to concept selection menu...")
                          return_to_menu = True # Set flag to break outer loop
                          break # Breaks out of the inner answer loop
                     elif lower_input == 'hint':
                          print("Generating hint...")
                          hint = get_ai_hint(model, selected_concept, question_text)
                          print(f"\nAI Tutor Hint:\n{hint}\n")
                          # Stay in this loop, prompt for answer again
                          continue
                     # Check for "I don't know" BEFORE processing as a potential answer
                     elif i_dont_know_pattern.match(lower_input):
                          print("Okay, let me try to help break that down...")
                          explanation = get_ai_scaffolded_explanation(model, selected_concept, question_text)
                          print(f"\nAI Tutor Explanation:\n{explanation}\n")
                          # Stay in this loop, prompt for answer again
                          continue
                     elif not user_input.strip(): # Handle empty answer after trying hint/menu/confusion
                          print("You didn't enter an answer. Please try again or use a command ('hint', 'menu', 'quit').")
                          continue
                     else:
                          # Valid answer provided (anything not a command or confusion phrase)
                          user_answer = user_input
                          break # Exit the inner answer loop to process the answer

                 except (EOFError, KeyboardInterrupt):
                      print("\nExiting tutor session. Goodbye!")
                      sys.exit(0)

            # Check the flag set by 'menu' command to break out of question loop
            if return_to_menu:
                 break # Break out of the question loop

            # --- Process the user's answer (only reached if user_input was NOT a command or "I don't know") ---
            print("Analyzing your answer...")
            feedback = get_ai_feedback(model, selected_concept, question_text, user_answer)

            print(f"\nAI Tutor Feedback:\n{feedback}\n")
            print("-" * 60) # Separator

        # --- End of questions for this concept ---
        # This block is reached if the question loop finishes OR if 'menu' was typed
        if not return_to_menu: # Only print end-of-concept if user didn't choose menu mid-concept
             print("\n" + "="*60)
             print(f"Completed Concept: {selected_concept.get('concept_name', 'Selected Concept')}")
             print(f"You've explored '{selected_concept.get('concept_name', 'this concept')}' focusing on {selected_concept.get('stem_misperception', 'bridging economic intuition')}. ")
             # Note for user: To address point 10 (question progression) and potentially connect concepts
             # add fields to your JSON like "next_concept_suggestion" or "key_takeaway"
             # and use them here.
             print("Returning to concept selection.")
             print("="*60 + "\n")


# --- Script Entry Point ---

if __name__ == "__main__":
    run_tutor()
