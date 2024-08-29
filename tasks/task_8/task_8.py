import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
        """
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
        """
        Initialize the Large Language Model (LLM) for quiz question generation.
        """
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.7,  # Example value; adjust as needed
            max_output_tokens=400  # Example value within recommended range
        )
        
    def generate_question_with_vectorstore(self):
        """
        Generate a quiz question using the topic provided and context from the vectorstore.
        """
        if not self.llm:
            raise ValueError("LLM is not initialized.")
        
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        
        # Retrieve relevant documents or context for the quiz topic from the vectorstore
        try:
            documents = self.vectorstore.similarity_search(self.topic)
        except AttributeError:
            raise ValueError("Vectorstore does not have a 'similarity_search' method.")
        
        if not documents:
            raise ValueError("No documents found for the given topic.")

        # Format the retrieved context and the quiz topic into a structured prompt
        prompt_template = PromptTemplate.from_template(self.system_template)
        formatted_prompt = prompt_template.format(topic=self.topic, context=' '.join(doc.page_content for doc in documents))
        
        # Generate the quiz question using the LLM
        try:
            response = self.llm.generate([formatted_prompt])  # Pass prompt as a list
        except ValueError as e:
            raise ValueError(f"Failed to generate question. Error: {str(e)}")
        
        # Extract the text from the LLMResult object
        question_str = response.generations[0][0].text  # Correctly accessing the first text result

        return question_str

    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.

        This method orchestrates the quiz generation process by utilizing the `generate_question_with_vectorstore` method to generate each question and the `validate_question` method to ensure its uniqueness before adding it to the quiz.

        Returns:
        - A list of dictionaries, where each dictionary represents a unique quiz question generated based on the topic.
        """
        self.question_bank = []  # Reset the question bank

        for _ in range(self.num_questions):
            # Generate a question string using the class method
            question_str = self.generate_question_with_vectorstore()

            # Convert the JSON string to a dictionary
            try:
                question = json.loads(question_str)
            except (json.JSONDecodeError, IndexError, KeyError):
                print("Failed to decode question JSON.")
                continue  # Skip this iteration if JSON decoding fails

            # Validate the question for uniqueness
            if self.validate_question(question):
                print("Successfully generated a unique question.")
                self.question_bank.append(question)  # Add the valid and unique question to the bank
            else:
                print("Duplicate or invalid question detected.")

        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        """
        Task: Validate a quiz question for uniqueness within the generated quiz.

        This method checks if the provided question (as a dictionary) is unique based on its text content compared to previously generated questions stored in `question_bank`.

        Returns:
        - A boolean value: True if the question is unique, False otherwise.
        """
        if "question" not in question:
            return False
        
        for existing_question in self.question_bank:
            if question["question"] == existing_question["question"]:
                return False  # Duplicate found
        
        return True


# Test the Object
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
    
        embed_client = EmbeddingClient(**embed_config)  # Initialize from Task 4
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
        
        # Add a method to retrieve the vectorstore
        def get_vectorstore(chroma_creator):
            # Return the Chroma collection (Chroma object) created in ChromaCollectionCreator
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
                
                # Retrieve the vectorstore from the creator
                vectorstore = get_vectorstore(chroma_creator)
                
                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, vectorstore)
                generator.init_llm()  # Initialize the LLM
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Questions: ")
            for question in question_bank:
                st.write(question)
