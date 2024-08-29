import streamlit as st
import os
import sys
import json
import re  # Import regex module to clean JSON strings
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator
from tasks.task_9.task_9 import QuizManager

# Helper function to initialize session state variables
def initialize_session_state():
    if 'question_bank' not in st.session_state:
        st.session_state['question_bank'] = []
    if 'question_index' not in st.session_state:
        st.session_state['question_index'] = 0
    if 'display_quiz' not in st.session_state:
        st.session_state['display_quiz'] = False

# Clean the JSON string by removing unwanted formatting
def clean_json_string(json_string):
    # Remove any ```json``` or other markdown-like wrapping
    cleaned_string = re.sub(r'```json|```', '', json_string).strip()
    return cleaned_string

# Step 1: Initialize session state variables
initialize_session_state()

if __name__ == "__main__":
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-explorer-423214",
        "location": "us-central1"
    }

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")

        # Initialize Document Processor and Embedding Client
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        # Step 2: Set topic input and number of questions
        with st.form("Load Data to Chroma"):
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Enter the quiz topic:")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=3)

            submitted = st.form_submit_button("Submit")

            if submitted:
                chroma_creator.create_chroma_collection()
                vectorstore = chroma_creator.db

                if vectorstore:
                    st.write(f"Generating {questions} questions for topic: {topic_input}")

                    # Step 3: Initialize QuizGenerator
                    generator = QuizGenerator(topic=topic_input, num_questions=questions, vectorstore=vectorstore)
                    generator.init_llm()
                    question_bank = []

                    for i in range(questions):
                        question_str = generator.generate_question_with_vectorstore()

                        # Debug: Output the raw JSON string
                        print(f"Generated question JSON: {question_str}")

                        # Clean the JSON string before attempting to decode it
                        question_str_cleaned = clean_json_string(question_str)
                        
                        try:
                            question = json.loads(question_str_cleaned)
                            question_bank.append(question)  # Only add valid questions
                        except json.JSONDecodeError as e:
                            print(f"Failed to decode question JSON: {e}")
                            continue

                    # Handle empty question bank
                    if len(question_bank) == 0:
                        st.error("No valid questions were generated. Please try again with a different topic or input.")
                    else:
                        # Step 4: Store the question bank in Streamlit's session state
                        st.session_state['question_bank'] = question_bank

                        # Step 5: Set display_quiz flag in session state
                        st.session_state['display_quiz'] = True

                        # Step 6: Set the question_index to 0
                        st.session_state['question_index'] = 0

    if st.session_state['display_quiz']:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Questions: ")

            # Initialize QuizManager with the question bank
            quiz_manager = QuizManager(st.session_state['question_bank'])

            # Ensure there are questions in the bank
            if quiz_manager.total_questions == 0:
                st.error("No questions available to display.")
            else:
                # Step 7: Set index_question
                index_question = quiz_manager.get_question_at_index(st.session_state['question_index'])

                # Format choices for radio button
                choices = [f"{choice['key']}) {choice['value']}" for choice in index_question['choices']]

                with st.form("MCQ"):
                    st.write(f"{st.session_state['question_index'] + 1}. {index_question['question']}")
                    answer = st.radio("Choose an answer", choices)

                    answer_choice = st.form_submit_button("Submit")

                    if answer_choice:
                        correct_answer_key = index_question['answer']
                        if answer.startswith(correct_answer_key):
                            st.success("Correct!")
                        else:
                            st.error("Incorrect!")
                        st.write(f"Explanation: {index_question['explanation']}")

                # Move navigation buttons outside the form
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Previous Question"):
                        st.session_state['question_index'] = (st.session_state['question_index'] - 1) % quiz_manager.total_questions
                        st.rerun()  # Refresh to show the previous question
                with col2:
                    if st.button("Next Question"):
                        st.session_state['question_index'] = (st.session_state['question_index'] + 1) % quiz_manager.total_questions
                        st.rerun()  # Refresh to show the next question
