import logging

from app.config.settings import get_settings
from app.documents.importer import DocumentImporter
from app.documents.loader import DocumentLoader
from app.exceptions.base import FaqBotError
from app.quality.classification.centroid_store import CentroidStore


logger = logging.getLogger(__name__)


class UpdateService:
    """Load source documents and pass them to the configured importer."""

    def __init__(
        self,
        loader: DocumentLoader,
        importer: DocumentImporter,
        centroid_store: CentroidStore | None = None,
    ) -> None:
        self.loader = loader
        self.importer = importer
        self.centroid_store = centroid_store

    async def update_database(self) -> None:
        """Load the configured FAQ input, import it, and refresh the centroids."""
        try:
            document = await self.loader.load(
                get_settings().FAQ_INPUT_FILE,
            )
            await self.importer.import_data(document)
            if self.centroid_store is not None:
                # Läuft im selben Hintergrund-Task wie der Import: die KB ist
                # konsistent, bevor Zentroide gebaut werden; danach gilt die
                # neue Wissensbasis inklusive Facetten vollständig.
                await self.centroid_store.rebuild(document.entries)

        except FaqBotError:
            raise
        except Exception:
            logger.exception("Error occurred while updating database")
            raise

    def get_database(self) -> dict[str]:
        """Return the importer's current vector-store contents."""
        return self.importer.get_current_database()