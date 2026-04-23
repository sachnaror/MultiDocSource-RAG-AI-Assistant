from app.services.embeddings import HashEmbeddingService
from app.services.source_registry import InMemoryRegistry
from app.services.vector_store import VectorStore
from app.workers.jobs import JobManager

registry = InMemoryRegistry()
embedder = HashEmbeddingService()
vector_store = VectorStore(embedder=embedder)
job_manager = JobManager(registry=registry)
job_manager.vector_store = vector_store
