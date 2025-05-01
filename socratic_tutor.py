# socratic_tutor.py

import json
import os
import sys
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv # <--- IMPORT ADDED

# --- Load Environment Variables ---
load_dotenv() # <--- LOAD .env FILE HERE, before accessing the key

# --- Configuration ---
QA_BANK_FILE = "qa_bank.json"
# NOTE: API Key will now be attempted to be loaded from a .env file first,
# then fallback to system environment variables if not found in .env
API_KEY = os.getenv("GEMINI_API_KEY") # <--- This line remains the same
MODEL_NAME = "gemini-1.5-flash-latest"

# --- Helper Functions ---

def load_qa_bank(filename):
    """Loads the question and answer bank from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "concepts" not in data or not isinstance(data["concepts"], list):
            print(f"Error: JSON file '{filename}' must contain a top-level key 'concepts' which is a list.", file=sys.stderr)
            sys.exit(1)
        return data["concepts"]
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
    """Configures the Gemini API and returns the model instance."""
    # This check now works correctly after load_dotenv() attempt
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
        print(f"Error configuring Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

def get_ai_feedback(model, concept, question, user_answer):
    """Gets feedback from the Gemini model on the user's answer."""

    # Construct a detailed prompt for the AI tutor
    prompt = f"""
    Context:
    You are an AI Economics Tutor interacting with an undergraduate STEM student.
    The student may have misconceptions based on over-reliance on physics, math, or engineering principles.
    The current economic concept being discussed is: "{concept['concept_name']}"
    A potential STEM-based misperception for this concept is: "{concept['stem_misperception']}"

    Task:
    Analyze the student's response below to the Socratic question provided.
    Provide concise (1-4 sentences) feedback. Focus on:
    - Acknowledging correct points (if any).
    - Gently identifying potential conceptual gaps or misunderstandings, especially those related to STEM analogies vs. economic reasoning.
    - Offering clarification or correction of basic economic principles involved.
    - Do NOT ask follow-up questions. Only provide commentary/feedback on the given answer.

    Socratic Question Asked:
    "{question}"

    Student's Answer:
    "{user_answer}"

    Your Feedback (as Economics Tutor):
    """

    try:
        # Set safety settings to allow potentially nuanced economic discussion
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5
            )
        )
        # Handle potential blocked responses
        if not response.parts:
             if response.prompt_feedback.block_reason:
                 return f"[AI Feedback Blocked: {response.prompt_feedback.block_reason}] Please try rephrasing your answer."
             else:
                 return "[AI Feedback Unavailable: No response part received, reason unknown]"

        return response.text.strip()

    except exceptions.GoogleAPIError as e:
        print(f"\nError communicating with Gemini API: {e}", file=sys.stderr)
        return "[AI Feedback Error: Could not connect to API]"
    except Exception as e:
        print(f"\nAn unexpected error occurred during AI feedback generation: {e}", file=sys.stderr)
        return "[AI Feedback Error: Unexpected issue]"

# --- Main Tutor Logic ---

def run_tutor():
    """Runs the main Socratic tutoring session."""
    concepts = load_qa_bank(QA_BANK_FILE)
    model = configure_gemini() # API Key check happens inside here now
    total_concepts = len(concepts)

    print("\n--- Welcome to the Socratic Economics Tutor for STEM Students ---")
    print("We will go through several economic concepts. For each, I will ask questions.")
    print("Please type your answer and press Enter. AI feedback will follow.")
    print("------------------------------------------------------------------\n")

    for i, concept in enumerate(concepts):
        concept_num = i + 1
        print(f"=== Concept {concept_num}/{total_concepts}: {concept['concept_name']} ===")
        print(f"Potential STEM Misconception Focus: {concept['stem_misperception']}\n")

        questions = concept.get("socratic_questions", [])
        total_questions = len(questions)

        if not questions:
            print("No questions found for this concept. Moving to the next one.\n")
            continue

        for j, question in enumerate(questions):
            question_num = j + 1
            print(f"-- Question {question_num}/{total_questions} --")
            print(f"Q: {question}")

            try:
                user_answer = input("Your Answer: ")
            except EOFError:
                print("\nExiting tutor session.")
                sys.exit(0)
            except KeyboardInterrupt:
                 print("\nExiting tutor session.")
                 sys.exit(0)


            if user_answer.strip().lower() in ['quit', 'exit']:
                 print("\nExiting tutor session.")
                 sys.exit(0)


            print("Analyzing your answer...")
            feedback = get_ai_feedback(model, concept, question, user_answer)

            print(f"\nAI Tutor Feedback:\n{feedback}\n")
            print("-" * 60) # Separator

        print(f"=== End of Concept {concept_num}: {concept['concept_name']} ===\n")

    print("--- You have completed all concepts. End of tutor session. ---")

# --- Script Entry Point ---

if __name__ == "__main__":
    run_tutor()