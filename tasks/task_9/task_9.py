import streamlit as st
import os
import sys
import json

sys.path.append(os.path.abspath('../../'))

from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.system_template = """
        You are a subject matter expert on the topic: {topic}

        Follow the instructions to create a quiz question:
        1. Generate a question based on the topic provided and context as key "question"
        2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
        3. Provide the correct answer for the question from the list of answers as key "answer"
        4. Provide an explanation as to why the answer is correct as key "explanation"

        You must respond as a JSON object with the following structure:
        {{
            "question": "<question>",
            "choices": [
                {{"key": "A", "value": "<choice>"}},
                {{"key": "B", "value": "<choice>"}},
                {{"key": "C", "value": "<choice>"}},
                {{"key": "D", "value": "<choice>"}}
            ],
            "answer": "<answer key from choices list>",
            "explanation": "<explanation as to why the answer is correct>"
        }}

        Context: {context}
        """

    def init_llm(self):
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.7,
            max_output_tokens=400
        )
        
    def generate_question_with_vectorstore(self):
        if not self.llm:
            raise ValueError("LLM is not initialized.")
        
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        
        try:
            documents = self.vectorstore.similarity_search(self.topic)
        except AttributeError:
            raise ValueError("Vectorstore does not have a 'similarity_search' method.")
        
        if not documents:
            raise ValueError("No documents found for the given topic.")

        prompt_template = PromptTemplate.from_template(self.system_template)
        formatted_prompt = prompt_template.format(topic=self.topic, context=' '.join(doc.page_content for doc in documents))
        
        try:
            response = self.llm.generate([formatted_prompt])
        except ValueError as e:
            raise ValueError(f"Failed to generate question. Error: {str(e)}")
        
        question_str = response.generations[0][0].text

        return question_str

    def generate_quiz(self) -> list:
        self.question_bank = []

        for _ in range(self.num_questions):
            question_str = self.generate_question_with_vectorstore()

            try:
                question = json.loads(question_str)
            except (json.JSONDecodeError, IndexError, KeyError):
                print("Failed to decode question JSON.")
                continue

            if self.validate_question(question):
                print("Successfully generated a unique question.")
                self.question_bank.append(question)
            else:
                print("Duplicate or invalid question detected.")

        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        if "question" not in question:
            return False
        
        for existing_question in self.question_bank:
            if question["question"] == existing_question["question"]:
                return False
        
        return True


class QuizManager:
    def __init__(self, questions: list):
        self.questions = questions
        self.total_questions = len(questions)

    def get_question_at_index(self, index: int):
        valid_index = index % self.total_questions
        return self.questions[valid_index]
    
    def next_question_index(self, direction=1):
        current_index = st.session_state.get("question_index", 0)
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index


if __name__ == "__main__":
    from tasks.task_3.task_3 import DocumentProcessor
    from tasks.task_4.task_4 import EmbeddingClient
    from tasks.task_5.task_5 import ChromaCollectionCreator

    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-explorer-423214",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
        
        def get_vectorstore(chroma_creator):
            return chroma_creator.db

        question_bank = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                vectorstore = get_vectorstore(chroma_creator)
                generator = QuizGenerator(topic_input, questions, vectorstore)
                generator.init_llm()
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            
            quiz_manager = QuizManager(question_bank)
            question_index = st.session_state.get("question_index", 0)
            
            with st.form("Multiple Choice Question"):
                index_question = quiz_manager.get_question_at_index(question_index)
                
                choices = []
                for choice in index_question['choices']:
                    choice_key = choice["key"]
                    choice_value = choice["value"]
                    choices.append(f"{choice_key}) {choice_value}")
                
                st.write(index_question['question'])
                
                answer = st.radio(
                    'Choose the correct answer',
                    choices
                )
                submit_button = st.form_submit_button("Submit")
                
                if submit_button:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")

            next_button = st.button("Next Question")
            prev_button = st.button("Previous Question")
            
            if next_button:
                quiz_manager.next_question_index(1)
                st.experimental_rerun()
                
            if prev_button:
                quiz_manager.next_question_index(-1)
                st.experimental_rerun()
