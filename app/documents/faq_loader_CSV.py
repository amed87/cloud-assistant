from app.documents.models import FAQEntry
from app.documents.loader import FAQLoader
from pathlib import Path
import hashlib
import json
import csv

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
        """Parst die FAQ-Einträge aus einer CSV-Datei."""
        with path.open(encoding=encoding) as f:
            reader = csv.DictReader(f)
            entries = []
            for row in reader:
                if not (row and row["ID"] and row["Themen"] and row["Schlüsselwörter"] and row["Frage"] and row["Antwort"] and row["Fragetyp"]):
                    continue
                subjects = row["Themen"].split()
                keywords = row["Schlüsselwörter"].split()
                version_hash = self._generate_entry_hash(
                    question=row["Frage"],
                    answer=row["Antwort"],
                    subjects=subjects,
                    question_type=row["Fragetyp"],
                    keywords=keywords,
                )
                entries.append(
                    FAQEntry(
                        id=f"{row['ID']}:{version_hash}",
                        question=row["Frage"],
                        answer=row["Antwort"],
                        subjects=subjects,
                        question_type=row["Fragetyp"],
                        keywords=keywords,
                    )
                )
            return entries