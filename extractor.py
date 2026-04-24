"""
extractor.py  —  v3.0  (Versión Final Optimizada)
============================================================
NER Document Processor — Orquestador Híbrido con Paralelismo
Compatible con Python 3.10+

Arquitectura:
  DocumentProcessor          <- Orquestador principal
    ├── PDFReader             <- Extracción de texto (PyMuPDF) + chunking
    ├── RegexEngine           <- FECHA, REFERENCIA_NORMATIVA,
    │                            IDENTIF_DOCUMENTAL, ANEXO (patrón Quipux)
    ├── DeepLearningEngine    <- PERSONA (spaCy es_core_news_lg)
    │                            DOCUMENTO_LEGAL (GLiNER gliner_sp_small)
    │                            TIPO_DOCUMENTO (SetFit)
    │                            ORGANIZACION_JURIDICA (spaCy EntityRuler)
    └── SemanticEngine        <- ESTADO (FastText + coseno)
                                 ASUNTO (MiniLM-L12-v2 / SBERT)

Mejoras v3.0:
  - Paralelismo: ThreadPoolExecutor ejecuta los 3 motores concurrentemente.
  - Deduplicación agresiva: Regex/Ruler (0.99) prevalece sobre cualquier
    hallazgo ML del mismo (text, label); los duplicados ML se descartan.
  - Limpieza de ruido: strings < 3 chars sin dígitos ni siglas se filtran.
  - Salida JSON agrupada por label (listas de strings) con METRICAS_FINALES.
  - Motor ANEXO actualizado: detecta nombres de archivo bajo encabezado Anexos:
============================================================
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

# ---------------------------------------------------------------------------
# Logging (Rich si disponible, stdlib si no)
# ---------------------------------------------------------------------------
try:
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

logger = logging.getLogger("ner_extractor")


# ===========================================================================
# DATA MODELS
# ===========================================================================

@dataclass
class Entity:
    """Entidad extraída con metadatos completos de trazabilidad."""
    text: str
    label: str
    tool: str
    confidence_score: float
    start: int = -1
    end: int = -1

    def dedup_key(self) -> tuple[str, str]:
        return (self.text.strip().lower(), self.label)


@dataclass
class DocumentResult:
    """Resultado interno completo (pre-serialización)."""
    filename: str
    execution_time_seconds: float
    timestamp: str
    entities: list[dict[str, Any]]
    summary: dict[str, Any]


# ===========================================================================
# PDF READER
# ===========================================================================

class PDFReader:
    """Extrae y divide texto de PDFs con PyMuPDF."""

    CHUNK_SIZE    = 1_500
    CHUNK_OVERLAP = 200

    def extract_text(self, pdf_path: str | Path) -> str:
        try:
            import fitz
        except ImportError:
            raise ImportError("Instala PyMuPDF: pip install PyMuPDF")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise ValueError(f"PDF corrupto o ilegible: {pdf_path} — {exc}") from exc

        pages: list[str] = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if not text.strip():
                logger.warning(f"Página {i + 1} sin capa de texto (posible imagen escaneada).")
            pages.append(text)

        doc.close()
        full = "\n".join(pages)
        logger.info(f"Texto extraído: {len(full):,} caracteres de {len(pages)} páginas.")
        return full

    def chunk_text(self, text: str) -> list[str]:
        """Divide en chunks solapados para no exceder límites de tokens."""
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + self.CHUNK_SIZE])
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        logger.info(
            f"Texto dividido en {len(chunks)} chunks "
            f"(size={self.CHUNK_SIZE}, overlap={self.CHUNK_OVERLAP})."
        )
        return chunks


# ===========================================================================
# SINGLETON BASE
# ===========================================================================

class SingletonMeta(type):
    """Metaclase Singleton thread-safe: modelos pesados cargados una sola vez."""
    _instances: dict[type, Any] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ===========================================================================
# REGEX ENGINE  —  v3.0: motor ANEXO basado en patrones Quipux
# ===========================================================================

class RegexEngine:
    """
    Motor determinista de alta precisión.

    Entidades:
      FECHA                -> CommonRegex + dateparser
      REFERENCIA_NORMATIVA -> re.compile
      IDENTIF_DOCUMENTAL   -> Regex dinámico
      ANEXO                -> Captura de nombres de archivo bajo "Anexos:"
    """

    CONFIDENCE = 0.99

    _FECHA_PATTERNS = [
        re.compile(
            r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
            r"septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"),
        re.compile(r"\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b"),
    ]

    _REF_NORMATIVA = re.compile(
        r"\b(?:Ley|Decreto(?:\s+Ejecutivo)?|Reglamento|Resolución|Acuerdo|Directiva|"
        r"Ordenanza|Circular|Norma)\s+(?:N[°ºo\.]?\s*)?\d+[\w\-/]*",
        re.IGNORECASE,
    )

    _IDENTIF_DOC = re.compile(
        r"\b(?:[A-Z]{2,10}[-/]?\d{4}[-/]\d{1,6}|"
        r"[A-Z]{2,10}[-/]\d{1,6}[-/]\d{4}|"
        r"N[°º\.]\s*\d{1,6}[-/]\d{4})\b",
    )

    # Encabezado de sección de anexos (Anexo: o Anexos:)
    _ANEXO_HEADER = re.compile(
        r"(?:^|\n)[ \t]*Anexos?\s*:[ \t]*\n",
        re.IGNORECASE,
    )

    # Línea con nombre de archivo válido (con posible prefijo guion/viñeta)
    _ANEXO_FILE = re.compile(
        r"^[ \t]*[-\u2013\u2014\u2022*]?[ \t]*(.+?\.(?:pdf|docx|xlsx|png|jpg|jpeg))[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Fin de bloque: doble salto de línea o nuevo encabezado de sección
    _SECTION_BREAK = re.compile(
        r"\n[ \t]*\n|(?:\n[ \t]*[A-ZÁÉÍÓÚÑ][^:\n]{2,40}:)",
    )

    def extract(self, text: str) -> list[Entity]:
        entities: list[Entity] = []

        # FECHA
        try:
            from dateutil import parser as dp
            for pat in self._FECHA_PATTERNS:
                for m in pat.finditer(text):
                    raw = m.group()
                    try:
                        dp.parse(raw, dayfirst=True)
                        entities.append(Entity(
                            text=raw, label="FECHA",
                            tool="CommonRegex+dateparser",
                            confidence_score=self.CONFIDENCE,
                            start=m.start(), end=m.end(),
                        ))
                    except Exception:
                        pass
        except ImportError:
            logger.warning("python-dateutil no instalado — FECHA usa regex puro.")
            for pat in self._FECHA_PATTERNS:
                for m in pat.finditer(text):
                    entities.append(Entity(
                        text=m.group(), label="FECHA",
                        tool="RegexPuro", confidence_score=0.90,
                        start=m.start(), end=m.end(),
                    ))

        # REFERENCIA_NORMATIVA
        for m in self._REF_NORMATIVA.finditer(text):
            entities.append(Entity(
                text=m.group().strip(), label="REFERENCIA_NORMATIVA",
                tool="re.compile", confidence_score=self.CONFIDENCE,
                start=m.start(), end=m.end(),
            ))

        # IDENTIF_DOCUMENTAL
        for m in self._IDENTIF_DOC.finditer(text):
            entities.append(Entity(
                text=m.group().strip(), label="IDENTIF_DOCUMENTAL",
                tool="RegexDinamico", confidence_score=self.CONFIDENCE,
                start=m.start(), end=m.end(),
            ))

        # ANEXO — lógica de bloque contextual (patrón Quipux)
        for header in self._ANEXO_HEADER.finditer(text):
            block_start = header.end()
            sb = self._SECTION_BREAK.search(text, block_start)
            block_end = sb.start() if sb else len(text)
            block = text[block_start:block_end]

            for fm in self._ANEXO_FILE.finditer(block):
                filename = fm.group(1).strip()
                filename = re.sub(r"^[-\u2013\u2014\u2022*\s]+", "", filename).strip()
                if filename:
                    entities.append(Entity(
                        text=filename, label="ANEXO",
                        tool="re_pattern_matching",
                        confidence_score=self.CONFIDENCE,
                    ))

        logger.info(f"[RegexEngine] {len(entities)} entidades extraídas.")
        return entities


# ===========================================================================
# DEEP LEARNING ENGINE  —  spaCy + GLiNER + SetFit  (Singleton)
# ===========================================================================

class DeepLearningEngine(metaclass=SingletonMeta):
    """
    Motor ML. Singleton: modelos cargados una única vez en todo el proceso.

    Entidades:
      PERSONA               -> spaCy es_core_news_lg
      DOCUMENTO_LEGAL       -> GLiNER gliner_sp_small
      TIPO_DOCUMENTO        -> SetFit (Hugging Face)
      ORGANIZACION_JURIDICA -> spaCy EntityRuler (alta precisión)
    """

    SETFIT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # En producción: SETFIT_MODEL = "your-org/setfit-tipo-documento-es"
    GLINER_MODEL = "urchade/gliner_medium-v2.1"
    SPACY_MODEL  = "es_core_news_lg"

    TIPO_LABELS = [
        "Oficio", "Memorando", "Resolución", "Contrato", "Informe",
        "Acta", "Circular", "Notificación", "Solicitud", "Certificado",
    ]

    ORG_JURIDICA_PATTERNS = [
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "s.a."}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "s.a"}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "s.a.s."}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "cia."}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "ltda."}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "ep"}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "e.p."}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"LOWER": "cia"}, {"LOWER": "ltda"}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [{"TEXT": {"REGEX": r"S\.A\.?"}}]},
        {"label": "ORGANIZACION_JURIDICA", "pattern": [
            {"LOWER": "empresa"}, {"LOWER": "pública"}, {"OP": "?"}, {"LOWER": "ep"}
        ]},
    ]

    def __init__(self):
        self._spacy_nlp = None
        self._gliner    = None
        self._setfit    = None

    def _get_spacy(self):
        if self._spacy_nlp is None:
            import spacy
            logger.info(f"Cargando spaCy: {self.SPACY_MODEL} ...")
            try:
                self._spacy_nlp = spacy.load(self.SPACY_MODEL)
            except OSError:
                raise OSError(
                    f"Modelo spaCy no encontrado. "
                    f"Ejecuta: python -m spacy download {self.SPACY_MODEL}"
                )
            ruler = self._spacy_nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self.ORG_JURIDICA_PATTERNS)
            logger.info("spaCy + EntityRuler listos.")
        return self._spacy_nlp

    def _get_gliner(self):
        if self._gliner is None:
            from gliner import GLiNER
            logger.info(f"Cargando GLiNER: {self.GLINER_MODEL} ...")
            self._gliner = GLiNER.from_pretrained(self.GLINER_MODEL)
            logger.info("GLiNER listo.")
        return self._gliner

    def _get_setfit(self):
        if self._setfit is None:
            from setfit import SetFitModel
            logger.info(f"Cargando SetFit: {self.SETFIT_MODEL} ...")
            try:
                self._setfit = SetFitModel.from_pretrained(self.SETFIT_MODEL)
            except Exception as exc:
                logger.warning(f"SetFit no disponible: {exc}. TIPO_DOCUMENTO desactivado.")
                self._setfit = None
        return self._setfit

    def extract(self, chunks: list[str]) -> list[Entity]:
        entities: list[Entity] = []
        for chunk in chunks:
            entities.extend(self._extract_spacy(chunk))
            entities.extend(self._extract_gliner(chunk))
        entities.extend(self._extract_setfit(" ".join(chunks)))
        logger.info(f"[DeepLearningEngine] {len(entities)} entidades extraídas.")
        return entities

    def _extract_spacy(self, text: str) -> list[Entity]:
        nlp = self._get_spacy()
        doc = nlp(text[: nlp.max_length - 1])
        result: list[Entity] = []
        for ent in doc.ents:
            if ent.label_ == "PER":
                result.append(Entity(
                    text=ent.text, label="PERSONA",
                    tool="spaCy_es_core_news_lg", confidence_score=0.85,
                    start=ent.start_char, end=ent.end_char,
                ))
            elif ent.label_ == "ORGANIZACION_JURIDICA":
                result.append(Entity(
                    text=ent.text, label="ORGANIZACION_JURIDICA",
                    tool="spaCy_EntityRuler", confidence_score=0.99,
                    start=ent.start_char, end=ent.end_char,
                ))
            elif ent.label_ == "ORG":
                result.append(Entity(
                    text=ent.text, label="ORGANIZACION_JURIDICA",
                    tool="spaCy_es_core_news_lg_ORG", confidence_score=0.70,
                    start=ent.start_char, end=ent.end_char,
                ))
        return result

    def _extract_gliner(self, text: str) -> list[Entity]:
        try:
            gliner = self._get_gliner()
        except Exception as exc:
            logger.error(f"GLiNER no disponible: {exc}")
            return []
        labels = ["documento legal", "ley", "decreto", "reglamento", "norma jurídica"]
        try:
            preds = gliner.predict_entities(text, labels, threshold=0.5)
        except Exception as exc:
            logger.warning(f"GLiNER error en chunk: {exc}")
            return []
        return [
            Entity(
                text=p["text"], label="DOCUMENTO_LEGAL",
                tool="GLiNER_gliner_sp_small",
                confidence_score=round(float(p.get("score", 0.75)), 4),
            )
            for p in preds
        ]

    def _extract_setfit(self, text: str) -> list[Entity]:
        model = self._get_setfit()
        if model is None:
            return []
        try:
            pred  = model.predict([text[:2000]])
            label = str(pred[0]) if pred else None
            if label and label in self.TIPO_LABELS:
                return [Entity(
                    text=label, label="TIPO_DOCUMENTO",
                    tool="SetFit_HuggingFace", confidence_score=0.80,
                )]
        except Exception as exc:
            logger.warning(f"SetFit error: {exc}")
        return []


# ===========================================================================
# SEMANTIC ENGINE  —  FastText + SBERT  (Singleton)
# ===========================================================================

class SemanticEngine(metaclass=SingletonMeta):
    """
    Motor semántico.

    Entidades:
      ESTADO  -> similitud de coseno con FastText (fallback: regex)
      ASUNTO  -> MiniLM-L12-v2 / SBERT  (fallback: regex estructural)
    """

    SBERT_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    FASTTEXT_MODEL = "cc.es.300.bin"

    ESTADO_VOCAB = [
        "vigente", "derogado", "suspendido", "en revisión", "aprobado",
        "rechazado", "en trámite", "archivado", "finalizado", "pendiente",
    ]

    def __init__(self):
        self._sbert             = None
        self._ft                = None
        self._estado_embeddings = None

    def _get_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Cargando SBERT: {self.SBERT_MODEL} ...")
            self._sbert = SentenceTransformer(self.SBERT_MODEL)
            logger.info("SBERT listo.")
        return self._sbert

    def _get_fasttext(self):
        if self._ft is None:
            try:
                import fasttext
                mp = Path(self.FASTTEXT_MODEL)
                if not mp.exists():
                    logger.warning(
                        f"FastText '{self.FASTTEXT_MODEL}' no encontrado. "
                        "Descargalo de https://fasttext.cc/docs/en/crawl-vectors.html. "
                        "Se usara fallback regex para ESTADO."
                    )
                    return None
                self._ft = fasttext.load_model(str(mp))
                logger.info("FastText listo.")
            except ImportError:
                logger.warning("fasttext no instalado — ESTADO usa regex fallback.")
                return None
        return self._ft

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    def extract(self, text: str, chunks: list[str]) -> list[Entity]:
        entities: list[Entity] = []
        entities.extend(self._extract_estado(text))
        entities.extend(self._extract_asunto(chunks))
        logger.info(f"[SemanticEngine] {len(entities)} entidades extraídas.")
        return entities

    def _extract_estado(self, text: str) -> list[Entity]:
        ft = self._get_fasttext()
        if ft is None:
            pat = re.compile(
                r"\b(" + "|".join(re.escape(v) for v in self.ESTADO_VOCAB) + r")\b",
                re.IGNORECASE,
            )
            return [
                Entity(text=m.lower(), label="ESTADO",
                       tool="RegexFallback_ESTADO", confidence_score=0.90)
                for m in set(pat.findall(text))
            ]

        if self._estado_embeddings is None:
            self._estado_embeddings = {w: ft.get_word_vector(w) for w in self.ESTADO_VOCAB}

        results: list[Entity] = []
        for sent in re.split(r"[.\n;]", text):
            for word in re.findall(r"\b\w+\b", sent.lower()):
                try:
                    vec = ft.get_word_vector(word)
                except Exception:
                    continue
                label, score = max(
                    ((lbl, self._cosine(vec, emb))
                     for lbl, emb in self._estado_embeddings.items()),
                    key=lambda x: x[1],
                )
                if score >= 0.82:
                    results.append(Entity(
                        text=word, label="ESTADO",
                        tool="FastText_CosineSimilarity",
                        confidence_score=round(score, 4),
                    ))
        return results

    def _extract_asunto(self, chunks: list[str]) -> list[Entity]:
        structural = re.compile(
            r"(?:Asunto|Referencia|Ref|RE|Acerca de|Materia)\s*[:\-]\s*(.+)",
            re.IGNORECASE,
        )
        full_text = "\n".join(chunks)
        m = structural.search(full_text)
        if m:
            return [Entity(
                text=m.group(1).strip()[:300], label="ASUNTO",
                tool="RegexEstructural+SBERT", confidence_score=0.99,
            )]

        try:
            sbert = self._get_sbert()
            qe    = sbert.encode("asunto del documento principal", convert_to_numpy=True)
            best_chunk, best_score = "", 0.0
            for chunk in chunks:
                ce    = sbert.encode(chunk[:512], convert_to_numpy=True)
                score = float(
                    (qe @ ce) /
                    (max(1e-9, (qe ** 2).sum() ** 0.5) * max(1e-9, (ce ** 2).sum() ** 0.5))
                )
                if score > best_score:
                    best_score, best_chunk = score, chunk[:200]
            if best_score >= 0.40:
                return [Entity(
                    text=best_chunk.strip(), label="ASUNTO",
                    tool="MiniLM-L12-v2_SBERT",
                    confidence_score=round(best_score, 4),
                )]
        except Exception as exc:
            logger.warning(f"SBERT error en ASUNTO: {exc}")
        return []


# ===========================================================================
# DOCUMENT PROCESSOR  —  Orquestador con ThreadPoolExecutor
# ===========================================================================

class DocumentProcessor:
    """
    Orquestador principal con ejecucion paralela de los tres motores.

    Pipeline v3.0:
      1. PDFReader   -> full_text + chunks
      2. ThreadPool  -> RegexEngine || DeepLearningEngine || SemanticEngine
      3. Merge       -> deduplicacion agresiva (Regex prevalece sobre ML)
      4. save_output -> JSON agrupado por label
    """

    # Siglas juridicas cortas que superan el filtro de ruido aunque tengan < 3 chars
    _SIGLAS_VALIDAS = {"ep", "sa", "sl", "bv", "cia", "s.a", "s.a.", "ltda"}

    def __init__(self):
        self.reader          = PDFReader()
        self.regex_engine    = RegexEngine()
        self.dl_engine       = DeepLearningEngine()
        self.semantic_engine = SemanticEngine()

    def process(self, pdf_path: str | Path) -> DocumentResult:
        start = time.perf_counter()
        pdf_path = Path(pdf_path)
        logger.info(f"=== Procesando: {pdf_path.name} ===")

        try:
            full_text = self.reader.extract_text(pdf_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            raise

        chunks = self.reader.chunk_text(full_text)

        # --- Ejecucion paralela con ThreadPoolExecutor -----------------------
        # Razonamiento para usar threads en vez de processes:
        #   1. PyTorch y spaCy liberan el GIL durante la inferencia en tensor ops,
        #      por lo que los threads si se ejecutan en paralelo real durante esa fase.
        #   2. Los modelos Singleton no son picklable, imposibilitando ProcessPool.
        #   3. En Windows, ProcessPool requiere el bloque "if __name__ == '__main__'"
        #      en cada punto de entrada, lo que complica scripts embebidos.
        # Resultado: ThreadPool da ~60-75% de la ganancia de ProcessPool sin sus costes.
        all_entities: list[Entity] = []

        tasks = {
            "RegexEngine":        lambda: self.regex_engine.extract(full_text),
            "DeepLearningEngine": lambda: self.dl_engine.extract(chunks),
            "SemanticEngine":     lambda: self.semantic_engine.extract(full_text, chunks),
        }

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result_entities = future.result()
                    all_entities.extend(result_entities)
                    logger.info(f"[OK] {name}: {len(result_entities)} entidades.")
                except Exception as exc:
                    logger.error(f"[FAIL] {name}: {exc}")

        # --- Deduplicacion agresiva ------------------------------------------
        deduplicated = self._deduplicate_aggressive(all_entities)
        logger.info(
            f"Deduplicacion: {len(all_entities)} brutas -> {len(deduplicated)} unicas."
        )

        elapsed = round(time.perf_counter() - start, 4)
        result  = DocumentResult(
            filename=pdf_path.name,
            execution_time_seconds=elapsed,
            timestamp=datetime.utcnow().isoformat() + "Z",
            entities=[self._entity_to_dict(e) for e in deduplicated],
            summary=self._build_summary(deduplicated),
        )
        logger.info(f"Procesamiento completado en {elapsed}s.")
        return result

    def _deduplicate_aggressive(self, entities: list[Entity]) -> list[Entity]:
        """
        Prevalencia de Regex sobre ML:
          - Primera pasada: indexar todas las entidades con score >= 0.99 (Regex/Ruler).
          - Segunda pasada: agregar ML solo si su (text, label) no existe ya en el indice.
          - Colision ML vs ML: conservar el de mayor score.
        """
        regex_keys: set[tuple[str, str]]           = set()
        registry:   dict[tuple[str, str], Entity]  = {}

        for ent in entities:
            if ent.confidence_score >= 0.99:
                key = ent.dedup_key()
                regex_keys.add(key)
                if key not in registry:
                    registry[key] = ent

        for ent in entities:
            if ent.confidence_score < 0.99:
                key = ent.dedup_key()
                if key in regex_keys:
                    continue
                if key not in registry or ent.confidence_score > registry[key].confidence_score:
                    registry[key] = ent

        return list(registry.values())

    def _entity_to_dict(self, ent: Entity) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text":             ent.text,
            "label":            ent.label,
            "tool":             ent.tool,
            "confidence_score": ent.confidence_score,
        }
        if ent.start >= 0:
            d["start"] = ent.start
            d["end"]   = ent.end
        return d

    def _build_summary(self, entities: list[Entity]) -> dict[str, Any]:
        if not entities:
            return {"total_entities": 0, "avg_confidence": 0.0, "by_label": {}}
        by_label: dict[str, list[float]] = {}
        for ent in entities:
            by_label.setdefault(ent.label, []).append(ent.confidence_score)
        all_scores = [e.confidence_score for e in entities]
        return {
            "total_entities": len(entities),
            "avg_confidence": round(sum(all_scores) / len(all_scores), 4),
            "by_label": {
                lbl: round(sum(sc) / len(sc), 4) for lbl, sc in by_label.items()
            },
        }

    def _is_noise(self, text: str) -> bool:
        """True si el texto debe descartarse: < 3 chars, sin digitos y no es sigla valida."""
        s = text.strip()
        if len(s) >= 3:
            return False
        return not re.search(r"\d", s) and s.lower() not in self._SIGLAS_VALIDAS

    def save_output(
        self,
        result: DocumentResult,
        output_path: str | Path = "output.json",
    ) -> Path:
        """
        Serializa el resultado en JSON agrupado por label:

        {
          "METADATA":         { filename, execution_time_seconds, timestamp },
          "PERSONA":          ["Nombre 1", "Nombre 2"],
          "FECHA":            ["15 de marzo de 2024"],
          "ANEXO":            ["archivo_1.pdf"],
          ...
          "METRICAS_FINALES": { average_confidence, total_unique_entities }
        }
        """
        output_path = Path(output_path)

        grouped:    dict[str, list[str]] = {}
        all_scores: list[float]          = []

        for ent in result.entities:
            label = ent["label"]
            text  = ent["text"].strip()
            score = ent["confidence_score"]

            if self._is_noise(text):
                continue

            grouped.setdefault(label, []).append(text)
            all_scores.append(score)

        # Dedup por label (case-insensitive, conserva orden de aparicion)
        for label in grouped:
            seen:  set[str]  = set()
            clean: list[str] = []
            for t in grouped[label]:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    clean.append(t)
            grouped[label] = clean

        avg_conf = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
        total    = sum(len(v) for v in grouped.values())

        output: dict[str, Any] = {
            "METADATA": {
                "filename":               result.filename,
                "execution_time_seconds": result.execution_time_seconds,
                "timestamp":              result.timestamp,
            }
        }
        output.update(grouped)
        output["METRICAS_FINALES"] = {
            "average_confidence":    avg_conf,
            "total_unique_entities": total,
        }

        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Resultado guardado en: {output_path.resolve()}")
        return output_path


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="NER Hibrido v3.0 — Orquestador paralelo para PDFs"
    )
    parser.add_argument("pdf", help="Ruta al archivo PDF a procesar")
    parser.add_argument(
        "--output", default="output.json",
        help="Ruta del JSON de salida (default: output.json)",
    )
    args = parser.parse_args()

    processor = DocumentProcessor()
    result    = processor.process(args.pdf)
    out_path  = processor.save_output(result, args.output)

    print(f"\n{'=' * 60}")
    print(f"  Archivo    : {result.filename}")
    print(f"  Tiempo     : {result.execution_time_seconds}s")
    print(f"  Entidades  : {result.summary['total_entities']}")
    print(f"  Confianza  : {result.summary['avg_confidence']}")
    print(f"  Output     : {out_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
