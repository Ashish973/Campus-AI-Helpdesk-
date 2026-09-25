from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,  Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from chunk import load_documents, split_documents



class embeddingmanager:
    """handles documents embedding generation using sentence trasnformer"""

    def __init__(self,model_name:str = "all-MiniLM-L6-v2"):

        """Initialize the embedding manager with a specified model name."""

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model {self.model_name}:{e}")
            raise

    def generate_embeddings(self,texts:list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts using the loaded model.

        Args:
            texts (list[str]): A list of strings to generate embeddings for."""

        if not self.model:
            raise ValueError("Model is not loaded. Please load the model before generating embeddings.")

        embeddings = self.model.encode(texts)

        print(f"Generated embeddings with shape : {embeddings.shape}")

        return embeddings


#embedding_manager = embeddingmanager()
#print(embedding_manager)

"intialize vector store chromadb for storing embeddings"

class vectorstore:
    """manages documents embedding in a chromadb vector store"""

    def __init__(
        self,
        collection_name: str = "text_documents_cosine",
        persist_directory: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store")
    ):
        """ Initialize the vector store with a specified 
        collection name: name of the chromadb
        persiste_directory : directory where the vector store will be persisted"""
    
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize chromadb client and collection"""

        try:
            #create chromadb persistent client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)

            #create collection if not exists

            self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
            "description": "Collection of text document embeddings",
            "hnsw:space": "cosine"
            }
            )       

            print(f"vector store initialized.collection :{self.collection_name}")
            print(f"existing document in collection :{self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self,documents:List[Any],embeddings:np.ndarray):
        """Add documents and embedding to the vector store"""

        if len(documents)!= len(embeddings):
            raise ValueError("The number of documents and embeddings must be the same.")
        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i ,(doc,embedding) in enumerate(zip(documents,embeddings)):

            #Generate a unique ID for each document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)

            #embedding is a numpy array, convert it to list for storage

            embeddings_list.append(embedding.tolist())


        #Add to collection
        try:
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = documents_text
            )

            print(f"successfully added {len(documents)} documents to the vector store.")

            print(f"Total documents in collection : {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

#vectorstore = vectorstore()
#print(vectorstore)



if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    texts = [doc.page_content for doc in chunks]

    embedding_mgr = embeddingmanager()
    embeddings = embedding_mgr.generate_embeddings(texts)
    v_store = vectorstore()
    v_store.add_documents(chunks, embeddings)




    
