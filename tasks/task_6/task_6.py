import sys
import os
import streamlit as st

sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator

def main():
    st.header("Quizzify")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-explorer-423214",
        "location": "us-central1"
    }
    
    # Initialize screen container
    screen = st.empty()  # Screen 1, ingest documents
    with screen.container():
        st.header("Quizzify")
        
        # Initialize DocumentProcessor and ingest documents
        doc_processor = DocumentProcessor()
        doc_processor.ingest_documents()  # Ensure you have a way to upload PDFs
        
        # Initialize EmbeddingClient
        embed_client = EmbeddingClient(
            model_name=embed_config["model_name"],
            project=embed_config["project"],
            location=embed_config["location"]
        )
        
        # Initialize ChromaCollectionCreator
        chroma_creator = ChromaCollectionCreator(
            processor=doc_processor,
            embed_model=embed_client
        )
        
        document = None  # Initialize document variable
        
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            # Streamlit widgets for user input
            topic_input = st.text_input("Enter Quiz Topic")
            num_questions = st.slider("Number of Questions", min_value=1, max_value=20, value=5)
            
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:
                # Create Chroma collection
                chroma_creator.create_chroma_collection()
                
                # Query Chroma collection
                document = chroma_creator.query_chroma_collection(topic_input)  # Assuming it returns the top document
                
    if document:
        screen.empty()  # Screen 2
        with st.container():
            st.header("Query Chroma for Topic, top Document: ")
            st.write(document)

if __name__ == "__main__":
    main()
