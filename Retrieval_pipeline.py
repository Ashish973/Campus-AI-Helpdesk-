from embedding import embeddingmanager,vectorstore

class RAGRetriever:
    #handles query based retrieval from vector store

    def __init__(self,vector_store:vectorstore = None,embedding_manager: embeddingmanager = None):
        """ Initialize the retriever
        Args:
            vector_store:containig documents embeddings
            embedding_manager: used for generating embeddings for query"""

        self.vector_store = vector_store() if vector_store is not None else vectorstore()
        self.embedding_manager = embedding_manager if embedding_manager is not None else embeddingmanager()


    def retrieve(self,query:str , top_k:int=5,score_threshold:float=0.0) -> list[dict]:


        """Retrieve relevant documents from the vector store based on the query.

        Args:
            query (str): The input query string.
            top_k (int): The number of top documents to retrieve. Default is 5.
            score_threshold (float): The minimum similarity score for a document to be considered relevant. Default is 0.0.

        returns:list of dictionaries containing the retrieved documents and their metadata."""

        print(f"Retrieving documents for query : {query}")
        
        # Generate embedding for the query

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]  
        """Get the first (and only) embedding"""


        "search for similar documents in the vector store"

        try:
            results = self.vector_store.collection.query(query_embeddings = [query_embedding.tolist()],n_results = top_k)

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i,(doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):

                    ##convert distance to similarity score uses cosine similarity

                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:

                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1

                        })

                print(f"Retrieved {len(retrieved_docs)} documents after filtering.")

            else:
                print("No documents found for the given query.")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        


rag_retreiver = RAGRetriever()
#rag_retreiver.retrieve("How does the attandance record maintain")




