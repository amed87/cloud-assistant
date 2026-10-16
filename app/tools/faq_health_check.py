import asyncio
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from contextlib import contextmanager

import numpy as np

from app.models.faq_health import (
    FAQHealthCheckReport,
    SelfRetrievalReport,
    SelfRetrievalFailure,
    DuplicateReport,
    DuplicatePair,
    RetrievalMarginReport,
    LowMarginQuery,
    EmbeddingHealthReport,
)
from app.quality.ranking.reranker import Reranker
from app.vectorstores.base import VectorStore
from app.vectorstores.temp_chroma import TempChromaStore

if TYPE_CHECKING:
    from app.embeddings.base import EmbeddingProvider
    from app.documents.faq_loader_csv import FAQLoaderCSV
    from app.documents.faq_importer import FAQImporter

logger = logging.getLogger(__name__)

@contextmanager
def suppress_httpx_logs(level=logging.ERROR):
    """Temporär httpx Logging unterdrücken (für QA-Checks)."""
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    
    old_httpx_level = httpx_logger.level
    old_httpcore_level = httpcore_logger.level
    
    httpx_logger.setLevel(level)
    httpcore_logger.setLevel(level)
    
    try:
        yield
    finally:
        httpx_logger.setLevel(old_httpx_level)
        httpcore_logger.setLevel(old_httpcore_level)


def print_progress(check_name: str, current: int, total: int):
    """Einfacher Progress in derselben Zeile."""
    pct = (current / total * 100) if total > 0 else 100.0
    line = f"\r[{check_name}] {current}/{total} ({pct:.0f}%)"
    sys.stdout.write(line)
    sys.stdout.flush()

def print_done(check_name: str, extra: str = ""):
    """Abschluss mit neuer Zeile."""
    line = f"\r[{check_name}] Done ✓ {extra}\n" if extra else f"\r[{check_name}] Done ✓\n"
    sys.stdout.write(line)
    sys.stdout.flush()

class FAQHealthCheckService:
    """Quality Assurance für Vector Store vor Updates."""

    def __init__(
        self,
        embedding_provider: 'EmbeddingProvider',
        loader: 'FAQLoaderCSV',
        create_temp_store: Callable[[], TempChromaStore],
        create_importer: Callable[[VectorStore], 'FAQImporter'],
        reranker: 'Reranker',
        live_progress: bool = True,
    ):
        self.embedding_provider = embedding_provider
        self.loader = loader
        self.create_temp_store = create_temp_store
        self.create_importer = create_importer
        self.reranker = reranker
        self.live_progress = live_progress

    async def run_qa(
        self,
        file_path: str,
    ) -> FAQHealthCheckReport:
        """Kompletter QA-Flow mit Progress Reporting."""
        
        with suppress_httpx_logs():
            faq_document = await self.loader.load(file_path)
            logger.info("Loaded %d FAQ entries from %s", len(faq_document.entries), file_path)

            temp_store = self.create_temp_store()
            
            try:
                if not faq_document.entries:
                    return FAQHealthCheckReport(
                        status="FAIL",
                        self_retrieval=SelfRetrievalReport(0, 0, 0.0, []),
                        duplicates=DuplicateReport(),
                        retrieval_margins=RetrievalMarginReport(),
                        embedding_health=EmbeddingHealthReport(),
                        messages=["Keine FAQ-Einträge vorhanden."],
                        checked_at=datetime.utcnow().isoformat(),
                    )

                importer = self.create_importer(temp_store)
                await importer.import_data(faq_document)
                logger.info("Imported %d entries into temp store", len(faq_document.entries))

                report = await self._run_qa_checks(temp_store, faq_document.entries)

                self._write_health_check_to_file(
                    result=report,
                    output_dir=Path(__file__).resolve().parents[3]
                    / "eval"
                    / "output",
                )


                logger.info("QA checks completed, status: %s", report.status)
                return report
                
            finally:
                await temp_store.close()
                logger.info("Cleaned up temporary store")

    async def _run_qa_checks(
        self,
        store: VectorStore,
        faq_entries,
    ) -> FAQHealthCheckReport:
        """Alle QA-Checks mit Progress Reporting."""

        # 1. Self-Retrieval
        self_ret = await self._check_self_retrieval(store, faq_entries)

        # 2. Retrieval Margins
        margin_report = await self._analyze_retrieval_margins_from_queries(store, faq_entries)

        # 3. Embedding Health
        embedding_report = await self._check_embedding_health_from_db(store, faq_entries)

        # 4. Duplicates
        dup_report = await self._detect_duplicates_in_store(store, faq_entries)

        # Aggregation & Status
        messages = []
        fails = []

        if not self_ret.passes():
            fails.append("self_retrieval")
            messages.append(
                f"Self-Retrieval Rate {self_ret.rate:.1%} unter Threshold "
                f"(benötigt: 95%). {len(self_ret.failed_ids)} Fragen fanden sich nicht selbst."
            )

        if not dup_report.passes():
            fails.append("duplicates")
            semantic_only_dupes = sum(1 for p in dup_report.duplicate_pairs if p.semantics_issue and not p.intent_separatable)
            intent_separable = sum(1 for p in dup_report.duplicate_pairs if p.intent_separatable)
            messages.append(
                f"Found {dup_report.affected_count} duplicate/similar entries "
                f"({semantic_only_dupes} semantically identical pairs without intent separation, "
                f"{intent_separable} pairs potentially separatable by Reranker)."
            )

        if not margin_report.passes():
            fails.append("margins")
            messages.append(f"{margin_report.below_threshold_count} Queries haben Margin < 0.1.")

        if not embedding_report.passes():
            fails.append("embeddings")
            messages.append(
                f"Embedding Health {embedding_report.health_rate:.1%} "
                f"(NaN: {embedding_report.nan_count}, Zero: {embedding_report.zero_count})."
            )

        status = "PASS" if not fails else "FAIL"

        return FAQHealthCheckReport(
            status=status,
            self_retrieval=self_ret,
            duplicates=dup_report,
            retrieval_margins=margin_report,
            embedding_health=embedding_report,
            messages=messages,
            checked_at=datetime.utcnow().isoformat(),
        )

    async def _check_self_retrieval(
        self,
        store: VectorStore,
        faq_entries,
    ) -> SelfRetrievalReport:
        """Prüft ob jede FAQ-Frage sich selbst im Top-5 findet."""

        failed_entries = []
        total = len(faq_entries)

        for i, entry in enumerate(faq_entries):
            if self.live_progress:
                print_progress("Self-Retrieval", i + 1, total)
            
            embedding = await self.embedding_provider.embed(entry.question)
            results = await store.search(embedding, limit=5)
            results = self.reranker.rerank(results, entry.question)

            found_ids = [r.id for r in results]

            if entry.id not in found_ids:
                failed_entries.append(SelfRetrievalFailure(
                    id=entry.id,
                    question=entry.question,
                    metadata={"answer": entry.answer, "subjects": entry.subjects}
                ))
        
        if self.live_progress:
            fail_count = len(failed_entries)
            msg = f"⚠ {fail_count} failed" if fail_count > 0 else ""
            print_done("Self-Retrieval", msg)

        return SelfRetrievalReport(
            total_documents=total,
            passed_count=total - len(failed_entries),
            rate=(total - len(failed_entries)) / total if total > 0 else 0.0,
            failed_entries=failed_entries[:20],
        )

    async def _analyze_retrieval_margins_from_queries(
        self,
        store: VectorStore,
        faq_entries,
    ) -> RetrievalMarginReport:
        """Margin zwischen Rank-1 und Rank-2 für jede FAQ-Frage."""

        margins = []
        low_margin_entries = []
        total = len(faq_entries)

        for i, entry in enumerate(faq_entries):
            if self.live_progress:
                print_progress("Margins", i + 1, total)
            
            embedding = await self.embedding_provider.embed(entry.question)
            results = await store.search(embedding, limit=2)
            results = self.reranker.rerank(results, entry.question)

            if len(results) < 2:
                continue

            margin = results[0].score - results[1].score
            margins.append(margin)

            if margin < 0.1:
                low_margin_entries.append(LowMarginQuery(
                    query_id=entry.id,
                    query_question=entry.question,
                    rank_1_id=results[0].id,
                    rank_1_question=results[0].metadata.get("answer", "N/A")[:100],
                    rank_1_score=results[0].score,
                    rank_2_id=results[1].id,
                    rank_2_question=results[1].metadata.get("answer", "N/A")[:100],
                    rank_2_score=results[1].score,
                    margin=margin,
                ))

        if self.live_progress:
            low_count = sum(1 for m in margins if m < 0.1)
            msg = f"⚠ {low_count} low margin" if low_count > 0 else ""
            print_done("Margins", msg)

        if not margins:
            return RetrievalMarginReport()

        threshold = 0.1
        below = sum(1 for m in margins if m < threshold)

        return RetrievalMarginReport(
            avg_margin=float(np.mean(margins)),
            min_margin=float(np.min(margins)),
            below_threshold_count=below,
            low_margin_entries=low_margin_entries[:20],
        )

    async def _detect_duplicates_in_store(
        self,
        store: VectorStore,
        faq_entries,
    ) -> DuplicateReport:
        """Pairwise L2-Distance mit Progress."""

        if len(faq_entries) < 2:
            return DuplicateReport()

        sample_size = min(len(faq_entries), 500)
        sampled_entries = np.random.choice(faq_entries, sample_size, replace=False)

        duplicate_pairs = []
        threshold = 0.5
        
        # Gesamtanzahl möglicher Paare für Progress-Berechnung
        total_pairs = sample_size * (sample_size - 1) // 2
        pair_count = 0

        for i in range(sample_size):
            emb_i = np.array(await self.embedding_provider.embed(sampled_entries[i].question))

            for j in range(i + 1, sample_size):
                pair_count += 1
                if self.live_progress:
                    print_progress("Duplicates", pair_count, total_pairs)
                
                emb_j = np.array(await self.embedding_provider.embed(sampled_entries[j].question))
                dist = np.linalg.norm(emb_i - emb_j)

                if dist < threshold:
                    # Analyse: Haben beide den gleichen Fragetyp?
                    type_1 = sampled_entries[i].question_type
                    type_2 = sampled_entries[j].question_type
                    
                    # Beide haben Fragetyp definiert?
                    both_have_type = bool(type_1 and type_2)
                    
                    # Gleicher Fragetyp?
                    same_intent = (both_have_type and type_1 == type_2) if both_have_type else None
                    
                    # Entscheidung: Kann Reranker dies trennen?
                    # - Nur wenn BEIDE Fragetyp haben UND verschiedene Typen
                    intent_separatable = (both_have_type and type_1 != type_2)
                    
                    # Semantisches Problem = distance zu niedrig (unabhängig vom Fragetyp)
                    semantics_issue = True
                    
                    # Notes für klare Diagnose
                    notes_parts = []
                    notes_parts.append(f"Distance too low ({dist:.3f})")
                    
                    if intent_separatable:
                        notes_parts.append(f"Different question_type ({type_1} ≠ {type_2}) - Reranker can separate")
                    elif both_have_type and same_intent:
                        notes_parts.append(f"Same question_type ({type_1}) - true duplicate")
                    else:
                        notes_parts.append("Missing question_type metadata")
                    
                    duplicate_pairs.append(DuplicatePair(
                        id_1=sampled_entries[i].id,
                        question_1=sampled_entries[i].question,
                        id_2=sampled_entries[j].id,
                        question_2=sampled_entries[j].question,
                        distance=float(dist),
                        notes="; ".join(notes_parts),
                        semantics_issue=semantics_issue,
                        intent_separatable=intent_separatable,
                    ))

                    if len(duplicate_pairs) >= 100:
                        break

            if len(duplicate_pairs) >= 100:
                break

        if self.live_progress:
            print_done("Duplicates", f"| Found {len(duplicate_pairs)} pairs")

        return DuplicateReport(
            duplicate_pairs=duplicate_pairs,
            affected_count=len(set(
                pid for pair in duplicate_pairs
                for pid in [pair.id_1, pair.id_2]
            )),
        )

    async def _check_embedding_health_from_db(
        self,
        store: VectorStore,
        faq_entries,
    ) -> EmbeddingHealthReport:
        """Embedding-Qualität aus gespeichertem DB-State."""

        db = store.get()
        total = len(db.get("ids", []))

        if total == 0:
            return EmbeddingHealthReport(0, 0, 0, 0, 0.0, [])

        sample_size = min(len(faq_entries), 100)
        nan_count = 0
        zero_count = 0
        extreme_count = 0
        unhealthy_ids = []

        for i, entry in enumerate(faq_entries[:sample_size]):
            if self.live_progress:
                print_progress("Embedding Health", i + 1, sample_size)
            
            embedding = await self.embedding_provider.embed(entry.question)
            emb = np.array(embedding)

            is_unhealthy = False
            if np.isnan(emb).any():
                nan_count += 1
                is_unhealthy = True
            elif np.allclose(emb, 0):
                zero_count += 1
                is_unhealthy = True
            elif np.any(np.abs(emb) > 100):
                extreme_count += 1
                is_unhealthy = True

            if is_unhealthy:
                unhealthy_ids.append(entry.id)

        if self.live_progress:
            unhealthy_total = nan_count + zero_count + extreme_count
            msg = f"⚠ {unhealthy_total} unhealthy" if unhealthy_total > 0 else ""
            print_done("Embedding Health", msg)

        healthy = sample_size - nan_count - zero_count - extreme_count
        health_rate = healthy / sample_size if sample_size > 0 else 1.0

        return EmbeddingHealthReport(
            total_vectors=total,
            nan_count=int(nan_count * (total / sample_size)) if sample_size > 0 else 0,
            zero_count=int(zero_count * (total / sample_size)) if sample_size > 0 else 0,
            extreme_values_count=int(extreme_count * (total / sample_size)) if sample_size > 0 else 0,
            health_rate=health_rate,
            unhealthy_ids=unhealthy_ids,
        )

    def _write_health_check_to_file(
        self,
        result: FAQHealthCheckReport,
        output_dir: Path,
    ) -> Path:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_dir
            / f"health_check_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
            )

        return output_file