import json
import os
import sys
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
QA_BANK_FILE = "qa_bank.json"
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest"

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
        # --- Add validation for individual concept structure ---
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
                 print(f"Warning: Concept '{concept['concept_name']}' (index {i}) has an empty 'socratic_questions' list.", file=sys.stderr)
                 # Decide if you want to exit or just warn - warning seems okay
                 # sys.exit(1)
            for j, question in enumerate(concept["socratic_questions"]):
                 if not isinstance(question, str) or not question.strip():
                      print(f"Error: Question {j+1} (index {j}) in Concept '{concept['concept_name']}' (index {i})'s 'socratic_questions' is invalid or empty.", file=sys.stderr)
                      sys.exit(1)
        # --- End validation ---

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
        # Catch any other unexpected file reading errors
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
        # Check specific API errors if possible
        if "blocked" in str(e).lower():
             print("Reason: API access might be blocked or restricted.", file=sys.stderr)
        sys.exit(1)


def get_ai_feedback(model, concept, question, user_answer):
    """
    Gets feedback from the Gemini model on the user's answer.
    Constructs a prompt tailored for STEM students and economic concepts.
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
    Provide concise feedback (aim for 1-4 sentences). Your feedback should:
    1. Acknowledge any correct aspects of their answer.
    2. Gently identify any conceptual gaps, misunderstandings, or points where STEM intuition might be misleading.
    3. Briefly clarify or correct basic economic principles relevant to their answer.
    4. Be encouraging and educational.
    - Do NOT ask follow-up questions. Focus solely on providing commentary/feedback on the *given* answer.

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
            generation_config=genai.types.GenerationConfig(temperature=0.5)
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
    The hint should guide the student without giving the answer away.
    """
    prompt = f"""
    Context:
    You are an AI Economics Tutor providing a hint to an undergraduate STEM student (Math, Engineering, or Physics) who is stuck on a Socratic question.
    The concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Provide a brief (1-3 sentences) hint for the Socratic question below.
    The hint should:
    - Gently nudge the student towards the correct economic perspective.
    - Avoid giving the direct answer.
    - Potentially reference the difference between STEM intuition and economic thinking for this concept.
    - Be encouraging.

    Socratic Question:
    "{question}"

    Your Hint (as Economics Tutor):
    """
    try:
        # Use slightly higher temperature for potentially more varied hints, but keep it low
        safety_settings = [ # Same safety settings as feedback
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(temperature=0.6) # Slightly higher temp for hints
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


# --- Main Tutor Logic ---

def run_tutor():
    """Runs the main Socratic tutoring session with concept selection and hints."""
    concepts = load_qa_bank(QA_BANK_FILE)
    model = configure_gemini()

    print("\n--- Welcome to the Socratic Economics Tutor for STEM Students ---")
    print("Bridging the gap between STEM intuition and economic reasoning.")
    print("------------------------------------------------------------------\n")

    while True: # Main loop for concept selection
        print("Choose a concept to explore:")
        for i, concept in enumerate(concepts):
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

                concept_index = int(choice) - 1 # Convert to 0-based index

                if 0 <= concept_index < len(concepts):
                    selected_concept = concepts[concept_index]
                    break # Valid selection, exit inner loop
                else:
                    print(f"Invalid number. Please choose between 1 and {len(concepts)}.")

            except (EOFError, KeyboardInterrupt):
                print("\nExiting tutor session. Goodbye!")
                sys.exit(0)
            except ValueError: # Should be caught by isdigit() but good practice
                 print("Invalid input. Please enter a number or 'quit'.")


        # --- Run session for the selected concept ---
        print(f"\n=== Exploring: {selected_concept.get('concept_name', 'Selected Concept')} ===")
        print(f"Potential STEM Misconception Focus: {selected_concept.get('stem_misperception', 'N/A')}\n")

        questions = selected_concept.get("socratic_questions", [])
        total_questions = len(questions)

        if not questions:
            print("No questions found for this concept. Returning to concept selection.\n")
            continue # Go back to the main concept selection loop

        for j, question in enumerate(questions):
            question_num = j + 1
            print(f"-- Question {question_num}/{total_questions} --")
            print(f"Q: {question}")

            # --- Inner loop for getting user answer or hint ---
            while True:
                 try:
                     user_input = input("Your Answer (type 'hint' for a hint, 'menu' to return to concepts, 'quit' to exit): ").strip().lower()

                     if user_input == 'quit':
                          print("\nExiting tutor session. Goodbye!")
                          sys.exit(0)
                     elif user_input == 'menu':
                          print("\nReturning to concept selection menu...")
                          break # Breaks out of the inner answer loop, then the question loop
                     elif user_input == 'hint':
                          print("Generating hint...")
                          hint = get_ai_hint(model, selected_concept, question)
                          print(f"\nAI Tutor Hint:\n{hint}\n")
                          # Stay in this loop, prompt for answer again
                          continue
                     elif not user_input: # Handle empty answer after trying hint/menu
                          print("You didn't enter an answer. Please try again or use a command ('hint', 'menu', 'quit').")
                          continue
                     else:
                          # Valid answer provided
                          user_answer = user_input # Assign the non-command input as the answer
                          break # Exit the inner answer loop to process the answer

                 except (EOFError, KeyboardInterrupt):
                      print("\nExiting tutor session. Goodbye!")
                      sys.exit(0)

            # If the user typed 'menu', the inner loop broke, and we check that break here
            if user_input == 'menu':
                 break # Break out of the question loop to return to concept selection

            # --- Process the user's answer (only reached if user_input was NOT 'menu') ---
            print("Analyzing your answer...")
            feedback = get_ai_feedback(model, selected_concept, question, user_answer)

            print(f"\nAI Tutor Feedback:\n{feedback}\n")
            print("-" * 60) # Separator

        # --- End of questions for this concept ---
        # This block is reached if the question loop finishes *or* if 'menu' was typed
        if user_input != 'menu': # Only print end-of-concept if user didn't choose menu mid-concept
             print(f"=== End of Concept: {selected_concept.get('concept_name', 'Selected Concept')} ===\n")
             # Loop continues back to concept selection menu automatically


# --- Script Entry Point ---

if __name__ == "__main__":
    run_tutor()
