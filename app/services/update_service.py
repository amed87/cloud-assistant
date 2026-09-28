import logging

from app.config.settings import get_settings
from app.documents.importer import DocumentImporter
from app.documents.loader import DocumentLoader


logger = logging.getLogger(__name__)


class UpdateService:
    def __init__(
        self,
        loader: DocumentLoader,
        importer: DocumentImporter,
    ) -> None:
        self.loader = loader
        self.importer = importer

    async def update_database(self) -> None:
        try:
            document = await self.loader.load(
                get_settings().FAQ_INPUT_FILE,
            )
            await self.importer.import_data(document)

        except Exception:
            logger.exception("Error occurred while updating database")
            raise

    def get_database(self) -> dict[str]:
        return self.importer.get_current_database()