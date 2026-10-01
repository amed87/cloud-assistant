from app.documents.models import FAQEntry
from app.documents.loader import FAQLoader
from app.exceptions.document import FAQLoadError
from pathlib import Path
import hashlib
import json
import csv
import logging

logger = logging.getLogger(__name__)

class FAQLoaderCSV(FAQLoader):
    @staticmethod
    def _generate_entry_hash(
        question: str,
        answer: str,
        subjects: list[str],
        question_type: str,
        keywords: list[str],
    ) -> str:
        content = {
            "question": question.strip(),
            "answer": answer.strip(),
            "subjects": sorted(value.strip() for value in subjects if value.strip()),
            "question_type": question_type.strip(),
            "keywords": sorted(value.strip() for value in keywords if value.strip()),
        }

        canonical = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    
    def _parse(
        self,
        path: Path,
        encoding: str,
    ) -> list[FAQEntry]:
        """Parse FAQ entries from a CSV file."""
        try:
            with path.open(encoding=encoding) as file:
                reader = csv.DictReader(file, delimiter=';')
                required_columns = {"ID", "Frage", "Antwort"}
                missing_columns = required_columns - set(reader.fieldnames or [])

                if missing_columns:
                    missing = ", ".join(sorted(missing_columns))
                    message = f"FAQ CSV is missing required columns: {missing}."
                    raise FAQLoadError(message)

                entries = []
                for row in reader:
                    question = row.get("Frage") or ""
                    answer = row.get("Antwort") or ""
                    entry_id = row.get("ID") or ""

                    if not (entry_id and question and answer):
                        logger.info(
                            "Skipping FAQ row with missing ID, question, or answer in %s.",
                            path,
                        )
                        continue

                    subjects = (row.get("Themen") or "").split()
                    keywords = (row.get("Schlüsselwörter") or "").split()
                    question_type = row.get("Fragetyp") or ""
                    version_hash = self._generate_entry_hash(
                        question=question,
                        answer=answer,
                        subjects=subjects,
                        question_type=question_type,
                        keywords=keywords,
                    )
                    entries.append(
                        FAQEntry(
                            id=f"{entry_id}:{version_hash}",
                            question=question,
                            answer=answer,
                            subjects=subjects,
                            question_type=question_type,
                            keywords=keywords,
                        )
                    )
            return entries
        except FAQLoadError as ex:
            logger.error("FAQ CSV loading failed: %s File: %s", ex.message, path)
            raise
        except (OSError, UnicodeError, csv.Error) as ex:
            logger.exception("Failed to read or parse FAQ CSV file: %s", path)
            raise FAQLoadError(
                "FAQ CSV file could not be read or parsed."
            ) from ex