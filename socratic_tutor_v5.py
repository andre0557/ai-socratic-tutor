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
MODEL_NAME = "gemini-1.5-flash-latest"  # Use pro model for better reasoning

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

def get_student_discipline():
    """
    Asks the student to select their primary STEM discipline.
    Returns the selected discipline string.
    Handles input validation and allows exit.
    """
    disciplines = ["Mathematics", "Engineering", "Physics", "Other STEM"] # Added Other STEM as an option
    print("\nBefore we start, please select your primary STEM discipline:")
    for i, discipline in enumerate(disciplines):
        print(f"{i + 1}. {discipline}")

    while True:
        try:
            choice = input(f"Your discipline ({'/'.join([d for d in disciplines])} number, or 'quit'): ").strip().lower()

            if choice == 'quit':
                print("Exiting tutor session. Goodbye!")
                sys.exit(0)

            if not choice.isdigit():
                print("Invalid input. Please enter a number or 'quit'.")
                continue

            discipline_index = int(choice) - 1 # Convert to 0-based index

            if 0 <= discipline_index < len(disciplines):
                selected_discipline = disciplines[discipline_index]
                print(f"Selected discipline: {selected_discipline}")
                return selected_discipline
            else:
                print(f"Invalid number. Please choose between 1 and {len(disciplines)}.")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting tutor session. Goodbye!")
            sys.exit(0)
        except ValueError:
             print("Invalid input. Please enter a number or 'quit'.")


def get_ai_feedback(model, concept, question, user_answer, discipline):
    """
    Gets feedback from the Gemini model on the user's answer.
    Enhanced prompt for comprehensive analysis, specific examples, and discipline-specific frameworks.
    """
    # Enhanced discipline-specific instructions
    if discipline == "Mathematics":
        discipline_guidance = """
        Mathematics-Specific Guidance:
        - Include at least one formal mathematical representation (function, equation, set notation, etc.) in your response
        - Connect economic concepts to specific mathematical frameworks such as:
          * Supply/demand as functions: S(p) = a + bp, D(p) = c - dp
          * Utility maximization as constrained optimization: max U(x,y) subject to px*x + py*y = m
          * Opportunity cost as the slope of a production possibilities frontier: dY/dX
          * Equilibrium as the solution to a system of equations
        - Use precise mathematical terminology (e.g., convexity, monotonicity, set theory, optimization)
        - Explain economic principles in terms of functions, derivatives, constraints, and solutions
        - Draw parallels to mathematical proofs, formal logic, or axiomatic systems where relevant
        
        IMPORTANT: Avoid overusing the concept of "subjective value" - use diverse economic principles and varied mathematical frameworks in your explanations.
        """
    elif discipline == "Engineering":
        discipline_guidance = """
        Engineering-Specific Guidance:
        - Frame economic concepts in terms of systems design, efficiency, and optimization
        - Use analogies to feedback systems, control theory, or resource allocation problems
        - Include specific engineering-relevant examples:
          * Resource allocation as a constrained design problem
          * Market equilibrium as a balanced system
          * Economic externalities as system design failures
          * Incentives as feedback mechanisms
        - Compare/contrast with engineering optimization problems (e.g., minimizing cost while meeting constraints)
        - Relate to real-world engineering trade-offs and design decisions
        - Use quantitative examples with specific numbers, much like engineering specifications
        
        IMPORTANT: Avoid overusing the concept of "subjective value" - use diverse economic principles and varied engineering frameworks in your explanations.
        """
    elif discipline == "Physics":
        discipline_guidance = """
        Physics-Specific Guidance:
        - Connect economic concepts to physical systems, equilibrium states, and dynamic processes
        - Use analogies to:
          * Conservation laws (and explain when they don't apply in economics)
          * Equilibrium concepts (stable, unstable, metastable states)
          * Statistical mechanics and emergent properties
          * Forces, potential fields, and gradient descent
        - Explicitly address deterministic vs. probabilistic approaches
        - Contrast economic equilibrium with physical equilibrium
        - Explain when superposition principles do/don't apply in economics
        - Use examples that relate to physical systems (e.g., particle interactions as market transactions)
        
        IMPORTANT: Avoid overusing the concept of "subjective value" - use diverse economic principles and varied physics frameworks in your explanations.
        """
    else:  # Other STEM
        discipline_guidance = """
        STEM-Specific Guidance:
        - Use general scientific frameworks like hypothesis testing, empirical validation, and model building
        - Provide quantitative examples with specific numbers and variables
        - Distinguish between models and reality, addressing limitations of economic models
        - Compare/contrast economic systems with other complex systems studied in STEM fields
        - Use data-oriented examples and quantifiable outcomes where possible
        
        IMPORTANT: Avoid overusing the concept of "subjective value" - use diverse economic principles and varied scientific frameworks in your explanations.
        """

    prompt = f"""
    Context:
    You are an AI Economics Tutor interacting with an undergraduate STEM student whose primary discipline is {discipline}.
    The student may have misconceptions based on over-reliance on principles learned in their STEM field, which don't always directly apply to economics.
    Your goal is to help them bridge the gap and understand the specific economic reasoning, **leveraging and contrasting with their background in {discipline}.**

    The current economic concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Analyze the student's response below to the Socratic question provided.
    Provide feedback (aim for 2-6 sentences) that is encouraging and educational. Your feedback should:
    
    1.  **CRITICAL: Methodically analyze ALL components of the student's answer before responding.**
        - First, list each distinct claim, concept, or idea in their answer (as a mental exercise)
        - For EACH identified component, determine if it's correct, partially correct, or incorrect
        - Address EVERY substantive point the student made, even if it seems tangential
        - NEVER ignore or overlook any part of their response
    
    2.  Only AFTER thoroughly analyzing their entire answer:
        - ONLY validate points that are explicitly correct using clear phrases that reflect their exact wording
        - If their answer contains no clear correct points, simply acknowledge what they've said without false validation
    
    3.  Gently identify any conceptual gaps, misunderstandings, or points where intuition from {discipline} might be misleading in economics.
    
    4.  **Provide at least one concrete, specific example with actual numbers or quantities** that illustrates the economic concept. For instance, instead of saying "prices would rise," say "prices might rise by 15-20%, similar to how [specific real-world example]."
    
    5.  Draw explicit connections or contrasts to concepts from {discipline}, using precise terminology from that field.
    
    6.  Include at least one mathematical or visual representation relevant to {discipline} students. For Mathematics students, include an actual equation, function, or formal notation that represents the economic concept.
    
    7.  IMPORTANT: Avoid repetitively focusing on "subjective value" across multiple answers. Use diverse economic principles and concepts. For each question, emphasize a different principle or framework.
    
    {discipline_guidance}

    Constraint:
    - Do NOT ask follow-up questions. Focus solely on providing commentary/feedback on the *given* answer. Prioritize clarity and relevance to a {discipline} background. **Do NOT infer understanding.**

    Socratic Question Asked:
    "{question}"

    Student's Answer:
    "{user_answer}"

    Your Feedback (as Economics Tutor, tailored for {discipline} student):
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
            generation_config=genai.types.GenerationConfig(temperature=0.4)
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


def get_ai_hint(model, concept, question, discipline):
    """
    Gets a hint from the Gemini model for a given concept and question, tailored by discipline.
    Critically, it avoids falsely validating when no answer has been given.
    """
    # Discipline-specific instructions for hints
    if discipline == "Mathematics":
        discipline_guidance = """
        For Mathematics students:
        - Suggest thinking about the problem using mathematical frameworks like:
          * Functions and their properties (e.g., utility functions, production functions)
          * Optimization problems with constraints
          * Systems of equations with equilibrium solutions
          * Set theory or logic for decision-making
        - Frame economic concepts in terms of mathematical operations or structures
        - Use precise mathematical terminology familiar to mathematics students
        - Suggest a simple equation or mathematical relationship that models the economic concept
        """
    elif discipline == "Engineering":
        discipline_guidance = """
        For Engineering students:
        - Suggest thinking about the problem using engineering frameworks like:
          * Systems design with inputs, outputs, and feedback loops
          * Resource allocation and efficiency optimization
          * Constraint satisfaction problems
          * Signal processing or control system analogies
        - Frame economic concepts in terms of designed systems and their behaviors
        - Use analogies to engineering problems they might be familiar with
        - Suggest quantitative approaches with measurable variables and outcomes
        """
    elif discipline == "Physics":
        discipline_guidance = """
        For Physics students:
        - Suggest thinking about the problem using physics frameworks like:
          * Equilibrium states and stability analysis
          * Statistical mechanics and emergent behaviors
          * Conservation principles (and when they don't apply)
          * Force and potential field analogies
        - Frame economic concepts in terms of physical systems and their behaviors
        - Highlight similarities and differences between physical and economic equilibria
        - Use analogies to physical phenomena they might be familiar with
        """
    else:  # Other STEM
        discipline_guidance = """
        For STEM students:
        - Suggest thinking about the problem using scientific frameworks like:
          * Hypothesis testing and empirical analysis
          * Model building with variables and relationships
          * System dynamics and feedback effects
          * Data-driven decision making
        - Frame economic concepts in terms of scientific principles
        - Use general quantitative reasoning approaches
        - Suggest specific measurable factors to consider
        """


    prompt = f"""
    Context:
    You are an AI Economics Tutor providing a hint to an undergraduate STEM student whose primary discipline is {discipline} and who is stuck on a Socratic question. **The student has not yet provided an answer to the question.**
    The concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Provide a brief (2-4 sentences) hint for the Socratic question below.
    The hint should:
    - **CRITICAL: Never begin with validating language or phrases that imply the student has already made progress or provided a correct starting point.**
    - **Start the hint with neutral, encouraging language or a direct suggestion on how to think about the problem.** Use openers such as "Think about...", "Consider...", "A helpful way to approach this is...", "Here's something to consider...", "To get started, think about...".
    - Gently nudge the student towards the correct economic perspective related to the question.
    - **Suggest thinking about a related concept or framework specifically from {discipline} and how it might be similar or different in economics.**
    - Provide a CONCRETE EXAMPLE with specific numbers, values, or quantities that illustrates the concept.
    - Avoid giving the direct answer.
    - Be encouraging without falsely validating.

    {discipline_guidance}

    Socratic Question:
    "{question}"

    Your Hint (as Economics Tutor, tailored for {discipline} student):
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

def get_ai_scaffolded_explanation(model, concept, question, discipline):
    """
    Provides a more structured explanation when the student indicates they are stuck or confused.
    Enhanced with concrete examples and a tiered approach to explanation, tailored by discipline.
    """
    # Discipline-specific instructions for scaffolded explanations
    if discipline == "Mathematics":
        discipline_guidance = """
        For Mathematics students:
        - Begin with formal definitions and axioms, similar to how mathematical concepts are introduced
        - Structure the explanation similarly to a mathematical proof or derivation, with clear logical steps
        - Use precise notation and formal relationships where possible
        - Provide a concrete numerical example that shows each step in the reasoning
        - Draw parallels to mathematical structures, functions, and optimization problems
        - Suggest a formal framework for analyzing the economic concept 
        """
    elif discipline == "Engineering":
        discipline_guidance = """
        For Engineering students:
        - Begin with a practical problem statement, similar to engineering problem framing
        - Structure the explanation like a system analysis, with inputs, processes, and outputs
        - Use examples involving resource constraints, efficiency metrics, and trade-offs
        - Provide a concrete numerical example with specifications and parameters
        - Draw parallels to design optimization, feedback systems, or resource allocation
        - Suggest practical applications or real-world contexts
        """
    elif discipline == "Physics":
        discipline_guidance = """
        For Physics students:
        - Begin with fundamental principles, similar to how physics laws are introduced
        - Structure the explanation like a physical system analysis, with states and interactions
        - Use examples involving equilibrium, forces/incentives, and system dynamics
        - Provide a concrete numerical example with measurable parameters
        - Draw parallels to physical concepts while highlighting key differences
        - Emphasize when economic "laws" differ from physical laws (e.g., lack of conservation)
        """
    else:  # Other STEM
        discipline_guidance = """
        For STEM students:
        - Begin with core principles and empirical observations
        - Structure the explanation like a scientific analysis with hypotheses and evidence
        - Use examples involving data, relationships between variables, and observable outcomes
        - Provide a concrete numerical example with measurable parameters
        - Draw parallels to general scientific methodology and model-building
        - Emphasize both the strengths and limitations of economic models
        """

    prompt = f"""
    Context:
    You are an AI Economics Tutor helping an undergraduate STEM student whose primary discipline is {discipline} and who is completely stuck or confused by a Socratic question.
    The student indicated they don't know the answer or are too confused to respond. Your goal is to provide foundational understanding to help them try again, without giving the answer away, **tailoring the explanation for their {discipline} background.**

    The concept being discussed is: "{concept.get('concept_name', 'N/A')}"
    A common potential STEM-based misperception for this concept is: "{concept.get('stem_misperception', 'N/A')}"

    Task:
    Provide a scaffolded explanation to help the student approach the specific Socratic question below, starting from foundational principles. Your explanation should:
    
    1. Acknowledge their difficulty in an encouraging way.
    
    2. Break down the core economic principle needed using THREE TIERS:
       - TIER 1: Define the core concept in basic terms using language familiar to {discipline} students
       - TIER 2: Provide a CONCRETE EXAMPLE with SPECIFIC NUMBERS that illustrates the concept
         * Use actual quantities, prices, percentages, or measurable values
         * For example, instead of "when price increases, quantity demanded decreases," say "if the price of coffee increases from $3 to $5, consumption might fall from 100 to 60 cups per day"
       - TIER 3: Connect to the specific question being asked
    
    3. Draw a parallel to a concept from {discipline}, explaining:
       - How it's similar (the connection point)
       - How it's different (the key distinction)
       - Using terminology and frameworks from {discipline}
    
    4. Suggest a first step in approaching the problem that leverages their {discipline} background.
    
    5. Avoid giving the direct answer to the original question.
    
    6. Keep your explanation to 5-8 sentences total, focusing on clarity and precision.
    
    7. IMPORTANT: Avoid overusing the concept of "subjective value" across multiple explanations. Focus on diverse economic principles relevant to this specific question.
    
    {discipline_guidance}

    Socratic Question the student is stuck on:
    "{question}"

    Your Scaffolded Explanation (as Economics Tutor, tailored for {discipline} student):
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
    """Runs the main Socratic tutoring session with discipline selection, concept selection, hints, and scaffolding."""
    concepts = load_qa_bank(QA_BANK_FILE)
    model = configure_gemini()
    discipline = get_student_discipline() # Get student discipline at the start

    print("\n--- Welcome to the Socratic Economics Tutor for STEM Students ---")
    print(f"Tailoring your experience for {discipline}.")
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
                          hint = get_ai_hint(model, selected_concept, question_text, discipline) # Pass discipline
                          print(f"\nAI Tutor Hint:\n{hint}\n")
                          # Stay in this loop, prompt for answer again
                          continue
                     # Check for "I don't know" BEFORE processing as a potential answer
                     elif i_dont_know_pattern.match(lower_input):
                          print("Okay, let me try to help break that down...")
                          explanation = get_ai_scaffolded_explanation(model, selected_concept, question_text, discipline) # Pass discipline
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
            feedback = get_ai_feedback(model, selected_concept, question_text, user_answer, discipline) # Pass discipline

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
