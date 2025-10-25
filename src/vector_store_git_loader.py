import os
import uuid
from typing import List, Any, Optional, Callable
import numpy as np
from git import Repo
import logging

from langchain_community.document_loaders.git import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from sentence_transformers import SentenceTransformer
import chromadb

CHROMADB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
COLLECTION_NAME = "git_documents"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME,status_callback: Optional[Callable] = None):
        """Initialize the embedding model.
        Args:
            model_name (str): The name of the embedding model to use.
            status_callback (Callable, optional): Function to call with status updates.
        """
        self.model_name = model_name
        self.status_callback = status_callback
        self.model = None
        self._load_model()
    

    def _update_status(self, message: str):
        """Update status via callback if available."""
        if self.status_callback:
            self.status_callback(message)
        logging.info(message)

    
    def _load_model(self):
        """Load the embedding model."""
        try:
            self._update_status(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._update_status("Embedding model loaded successfully.")
        except Exception as e:
            self._update_status(f"Error loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        Args:
            texts (List[str]): The list of texts to embed.
        Returns:
            np.ndarray: The generated embeddings.
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")

        self._update_status(f"Generating embeddings for {len(texts)} texts.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self._update_status(f"Embeddings with shape {embeddings.shape} generated successfully.")
        return embeddings
    
class VectorStoreManager:
    """Manages the vector store using ChromaDB."""
    def __init__(self, collection_name: str = COLLECTION_NAME, db_dir: str = CHROMADB_DIR,status_callback: Optional[Callable] = None):
        """Initialize the vector store manager.
        Args:
            collection_name (str): The name of the collection in ChromaDB.
            db_dir (str): The directory to store the ChromaDB database.
            status_callback (Callable, optional): Function to call with status updates.
        """
        self.collection_name = collection_name
        self.db_dir = db_dir
        self.status_callback = status_callback
        self.client = None
        self.collection = None
        self._initialize_db()

    def _update_status(self, message: str):
        """Update status via callback if available."""
        if self.status_callback:
            self.status_callback(message)
        logging.info(message)
    
    def _initialize_db(self):
        """Initialize the ChromaDB client and collection."""
        try:
            ## Create directory if it doesn't exist
            os.makedirs(self.db_dir, exist_ok=True)
            self._update_status(f"Initializing ChromaDB at {self.db_dir}")
            self.client = chromadb.PersistentClient(path=self.db_dir)

            ## Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Collection of git document embeddings"}
            )
            self._update_status(f"Collection '{self.collection_name}' initialized successfully.")
            self._update_status(f"Current collections: {self.collection.count()}")
        except Exception as e:
            self._update_status(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store.
        Args:
            documents (List[Any]): The list of documents to add.
            embeddings (np.ndarray): The corresponding embeddings.
        """
        if not self.collection:
            raise ValueError("ChromaDB collection is not initialized.")
        
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")
        
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        # Preserve ALL metadata from the original documents, not just "source"
        metadatas = []
        for doc in documents:
            # Copy all metadata, but ensure all values are strings, numbers, or booleans
            # ChromaDB doesn't accept complex types in metadata
            metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
            # Ensure there's always a source field for compatibility
            if "source" not in metadata:
                metadata["source"] = "unknown"
            metadatas.append(metadata)
        
        texts = [doc.page_content for doc in documents]

        self._update_status(f"Adding {len(documents)} documents to the vector store.")
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=texts
            )
            self._update_status("Documents added successfully.")
            self._update_status(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            self._update_status(f"Error adding documents to ChromaDB: {e}")
            raise

    def as_retriever(self, k: int = 4):
        """Return a simple retriever that exposes get_relevant_documents(query).

        This wrapper returns an object compatible with LangChain retriever usage
        (i.e. has get_relevant_documents). It queries the underlying chromadb
        collection and converts results to LangChain Document objects.
        """
        if not self.collection:
            raise ValueError("ChromaDB collection is not initialized.")

        class _ChromaRetriever:
            def __init__(self, collection, k):
                self.collection = collection
                self.k = k

            def get_relevant_documents(self, query: str) -> List[Document]:
                # chromadb returns nested lists for documents/metadatas when passing a
                # list of query_texts. We query a single text so take index 0.
                try:
                    result = self.collection.query(
                        query_texts=[query],
                        n_results=self.k,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as e:
                    logging.error(f"Error querying chromadb collection: {e}")
                    raise

                docs = []
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                for text, meta in zip(documents, metadatas):
                    docs.append(Document(page_content=text, metadata=meta))
                return docs

        return _ChromaRetriever(self.collection, k)
    

class GitDocument:
    """Handles loading and processing documents from a Git repository, and storing them in a vector store."""
    def __init__(self, repo_url: str, local_path: str, vector_store: VectorStoreManager, embedding_manager: EmbeddingManager,status_callback: Optional[Callable] = None):
        """Initialize the GitDocument handler.
        Args:
            repo_url (str): The URL of the Git repository.
            local_path (str): The local path to clone the repository.
            vector_store (VectorStoreManager): The vector store manager instance.
            embedding_manager (EmbeddingManager): The embedding manager instance.
            status_callback (Callable, optional): Function to call with status updates.
        """
        self.repo_url = repo_url
        # sanitize and normalize local_path to avoid invalid characters/trailing spaces
        if isinstance(local_path, str):
            lp = local_path.strip()
            # replace trailing dots/spaces which can be invalid on Windows
            while lp.endswith(".") or lp.endswith(" "):
                lp = lp[:-1]
            self.local_path = os.path.normpath(lp)
        else:
            self.local_path = local_path
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.status_callback = status_callback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    
    def _update_status(self, message: str):
        """Update status via callback if available."""
        if self.status_callback:
            self.status_callback(message)
        logging.info(message)


    def _initialize_manager(self):
        """Initialize embedding and vector store managers if not already initialized."""
        if not self.embedding_manager:
            self.embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
        if not self.vector_store:
            self.vector_store = VectorStoreManager(collection_name=COLLECTION_NAME, db_dir=CHROMADB_DIR)
    
    def _setup_repo(self):
        """Clone the Git repository if it doesn't exist locally or pull the latest changes if it does."""
        # If the path exists, try to open the repo and pull latest changes.
        if os.path.exists(self.local_path):
            self._update_status(f"Local repository found at {self.local_path}; pulling latest changes")
            try:
                repo = Repo(self.local_path)
                origin = repo.remotes.origin
                origin.pull()
                self._update_status("Repository updated successfully.")
            except Exception as e:
                self._update_status(f"Error pulling repository: {e}")
                raise
        else:
            # Path doesn't exist: clone the repository into the target path.
            self._update_status(f"Cloning repository from {self.repo_url} to {self.local_path}")
            try:
                parent_dir = os.path.dirname(os.path.abspath(self.local_path)) or "."
                os.makedirs(parent_dir, exist_ok=True)
                Repo.clone_from(self.repo_url, self.local_path)
                self._update_status("Repository cloned successfully.")
            except Exception as e:
                self._update_status(f"Error cloning repository: {e}")
                raise
    

    def _load_documents(self):
        """Load documents from the Git repository using GitLoader.
        Returns:
            List[Any]: The list of loaded documents.
        """
        try:
            self._update_status(f"Loading documents from repository at {self.local_path}")
            loader = GitLoader(repo_path=self.local_path, branch="main")
            documents = loader.load()
            self._update_status(f"{len(documents)} documents loaded successfully.")
            return documents
        except Exception as e:
            self._update_status(f"Error loading documents: {e}")
            raise
    
    def process_and_store_documents(self):
        """Process documents from the Git repository and store them in the vector store."""
        self._initialize_manager()
        self._setup_repo()
        documents = self._load_documents()
        
        if not documents:
            logging.warning("No documents found to process.")
            return

        self._update_status("Splitting documents into chunks.")
        texts = self.text_splitter.split_documents(documents)
        self._update_status(f"{len(texts)} text chunks created.")

        embeddings = self.embedding_manager.generate_embeddings([doc.page_content for doc in texts])
        
        self.vector_store.add_documents(texts, embeddings)
        self._update_status("All documents processed and stored successfully.")


#### Example usage for testing purposes
def main():
    logging.basicConfig(level=logging.INFO)

    pipeline = GitDocument(
        repo_url="https://github.com/tharun067/THARUN-PORTFOLIO.git",
        local_path="./cloned_repo",
        vector_store=VectorStoreManager(),
        embedding_manager=EmbeddingManager(),
    )
    #pipeline.process_and_store_documents()
    logging.info("Process completed.")

    ### Example of using the retriever
    retriever = pipeline.vector_store.as_retriever(k=6)
    query = "what constants folder contains?"
    docs = retriever.get_relevant_documents(query)
    for i, d in enumerate(docs, 1):
        print(f"Result {i}: source={d.metadata.get('source')}, snippet={d.page_content[:200]!r}")
if __name__ == "__main__":
    main()

