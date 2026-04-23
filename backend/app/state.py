from app.core.config import VECTOR_BACKEND
from app.services.embeddings import HashEmbeddingService
from app.services.pinecone_store import PineconeVectorStore
from app.services.source_registry import InMemoryRegistry
from app.services.vector_store import VectorStore
from app.workers.jobs import JobManager

registry = InMemoryRegistry()
embedder = HashEmbeddingService()
vector_store: VectorStore
if VECTOR_BACKEND == "pinecone":
    vector_store = PineconeVectorStore(embedder=embedder)
else:
    vector_store = VectorStore(embedder=embedder)
job_manager = JobManager(registry=registry)
job_manager.embedder = embedder
job_manager.vector_store = vector_store
