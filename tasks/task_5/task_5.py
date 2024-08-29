import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient

# Import Task libraries
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        """
        Initializes the ChromaCollectionCreator with a DocumentProcessor instance and embeddings configuration.
        :param processor: An instance of DocumentProcessor that has processed documents.
        :param embed_model: An embedding client for embedding documents.
        """
        self.processor = processor      # This will hold the DocumentProcessor from Task 3
        self.embed_model = embed_model  # This will hold the EmbeddingClient from Task 4
        self.db = None                  # This will hold the Chroma collection
    
    def create_chroma_collection(self):
        """
        Create a Chroma collection from the documents processed by the DocumentProcessor instance.

        Steps:
        1. Check if any documents have been processed by the DocumentProcessor instance. If not, display an error message using Streamlit's error widget.
        2. Split the processed documents into text chunks suitable for embedding and indexing using CharacterTextSplitter.
        3. Create a Chroma collection in memory with the text chunks obtained from step 2 and the embeddings model initialized in the class.
        """
        # Step 1: Check for processed documents
        if not self.processor.pages:
            st.error("No documents found!", icon="🚨")
            return

        # Step 2: Split documents into text chunks
        texts = []
        splitter = CharacterTextSplitter(
            separator="\n",  # Using newline as a separator
            chunk_size=500,  # Define chunk size (e.g., 500 characters)
            chunk_overlap=100  # Define chunk overlap (e.g., 100 characters)
        )

        for page in self.processor.pages:
            # Use 'page_content' to get the document text
            document_text = getattr(page, 'page_content', '')
            if not document_text:
                st.error("Unable to access document content!", icon="🚨")
                return
            # Create Document objects from text chunks
            chunks = splitter.split_text(document_text)
            for chunk in chunks:
                texts.append(Document(page_content=chunk))

        if texts:
            st.success(f"Successfully split pages into {len(texts)} chunks!", icon="✅")
        else:
            st.error("Failed to split pages into chunks!", icon="🚨")
            return

        # Step 3: Create the Chroma Collection
        try:
            self.db = Chroma.from_documents(texts, self.embed_model.client)
            st.success("Successfully created Chroma Collection!", icon="✅")
        except Exception as e:
            st.error(f"Failed to create Chroma Collection! Error: {str(e)}", icon="🚨")
    
    def query_chroma_collection(self, query) -> Document:
        """
        Queries the created Chroma collection for documents similar to the query.
        :param query: The query string to search for in the Chroma collection.
        :return: The first matching document from the collection with similarity score.
        """
        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")

if __name__ == "__main__":
    processor = DocumentProcessor()  # Initialize from Task 3
    processor.ingest_documents()

    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-explorer-423214",
        "location": "us-central1"
    }

    embed_client = EmbeddingClient(**embed_config)  # Initialize from Task 4

    chroma_creator = ChromaCollectionCreator(processor, embed_client)

    with st.form("Load Data to Chroma"):
        st.write("Select PDFs for Ingestion, then click Submit")

        submitted = st.form_submit_button("Submit")
        if submitted:
            chroma_creator.create_chroma_collection()
