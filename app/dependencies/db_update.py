from app.documents.faq_loader_csv import FAQLoaderCSV
from app.documents.faq_importer import FAQImporter
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.services.update_service import UpdateService
from app.vectorstores.chroma import ChromaVectorStore

faq_loader = FAQLoaderCSV()
embedding_provider =  OllamaEmbeddingProvider()
vector_store = ChromaVectorStore()
faq_importer = FAQImporter(
    embedding_provider=embedding_provider,
    vector_store=vector_store
)
update_service = UpdateService(
    loader=faq_loader,
    importer=faq_importer
)

def get_update_service() -> UpdateService:
    return update_service