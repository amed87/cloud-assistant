from app.documents.models import FAQEntry
from app.documents.loader import FAQLoader
from pathlib import Path
import csv


class FAQLoaderCSV(FAQLoader):
    def _parse(
        self,
        path: Path,
        encoding: str,
    ) -> list[FAQEntry]:
        """Parst die FAQ-Einträge aus einer CSV-Datei."""
        with path.open(encoding=encoding) as f:
            reader = csv.DictReader(f)
            return [FAQEntry(id=str(index), question=row["Frage"], answer=row["Antwort"], keywords=row["Schlüsselwörter"].split(" ")) for index, row in enumerate(reader)]