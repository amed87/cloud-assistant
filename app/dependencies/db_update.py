from app.config.settings import get_settings
from app.documents.faq_loader_csv import FAQLoaderCSV
from app.documents.faq_importer import FAQImporter
from app.dependencies.providers import embedding_provider
from app.services.update_service import UpdateService
from app.vectorstores.chroma import ChromaVectorStore
from app.dependencies.centroids import centroid_store

faq_loader = FAQLoaderCSV()
vector_store = ChromaVectorStore()
faq_importer = FAQImporter(
    embedding_provider=embedding_provider,
    vector_store=vector_store
)
update_service = UpdateService(
    loader=faq_loader,
    importer=faq_importer,
    centroid_store=centroid_store,
)

def get_update_service() -> UpdateService:
    return update_service