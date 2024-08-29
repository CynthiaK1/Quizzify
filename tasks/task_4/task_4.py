import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings

import sys
print(sys.executable)
class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.

    The EmbeddingClient class should be capable of initializing an embedding client with specific configurations
    for model name, project, and location. Your task is to implement the __init__ method based on the provided
    parameters. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.
    """

    def __init__(self, model_name, project, location):
        # Initialize the VertexAIEmbeddings client with the given parameters
        self.client = VertexAIEmbeddings(
            model_name=model_name,
            project=project,
            location=location
        )

    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.

        :param query: The text query to embed.
        :return: The embeddings for the query or None if the operation fails.
        """
        vectors = self.client.embed_query(query)
        return vectors

    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.

        :param documents: A list of text documents to embed.
        :return: A list of embeddings for the given documents.
        """
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            st.write("Method embed_documents not defined for the client.")
            return None

def main():
    st.title("VertexAI Embeddings with Streamlit")
    st.write("Enter a query to get embeddings:")

    # Initialize the EmbeddingClient with your parameters
    model_name = "textembedding-gecko@003"
    project = "gemini-explorer-423214"
    location = "us-central1"
    embedding_client = EmbeddingClient(model_name, project, location)

    # Text input for the query
    query = st.text_input("Query:", "Hello World!")

    if st.button("Get Embeddings"):
        # Get embeddings for the query
        vectors = embedding_client.embed_query(query)
        if vectors:
            st.write("Embeddings:")
            st.write(vectors)
        else:
            st.write("Failed to get embeddings.")

if __name__ == "__main__":
    main()
