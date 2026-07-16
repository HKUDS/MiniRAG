import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from minirag.base import DocProcessingStatus, DocStatus
from minirag.minirag import MiniRAG


class FakeDocStatusStorage:
    def __init__(self, docs_by_status=None):
        self.docs_by_status = docs_by_status or {}
        self.upserts = []

    async def get_docs_by_status(self, status):
        return self.docs_by_status.get(status, {})

    async def upsert(self, data):
        self.upserts.append(data)


class FakeStorage:
    def __init__(self):
        self.upserts = []

    async def upsert(self, data):
        self.upserts.append(data)


def make_status_doc(content="A document"):
    now = datetime.now().isoformat()
    return DocProcessingStatus(
        content=content,
        content_summary=content,
        content_length=len(content),
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


def make_chunk(content="A document"):
    return {
        "tokens": 2,
        "content": content,
        "chunk_order_index": 0,
        "full_doc_id": "doc-1",
    }


def make_bare_rag(doc_status):
    rag = object.__new__(MiniRAG)
    rag.doc_status = doc_status
    rag.chunk_entity_relation_graph = object()
    rag.entities_vdb = object()
    rag.entity_name_vdb = object()
    rag.relationships_vdb = object()
    rag.chunk_overlap_token_size = 0
    rag.chunk_token_size = 100
    rag.tiktoken_model_name = "gpt-4o-mini"
    rag.max_parallel_insert = 2
    rag.chunks_vdb = FakeStorage()
    rag.full_docs = FakeStorage()
    rag.text_chunks = FakeStorage()
    rag._insert_done = AsyncMock()
    return rag


class IncrementalIndexingTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_insert_does_not_extract_processed_documents(self):
        existing_doc = make_status_doc("Already indexed")
        doc_status = FakeDocStatusStorage(
            {DocStatus.PROCESSED: {"doc-existing": existing_doc}}
        )
        rag = make_bare_rag(doc_status)
        rag.chunking_func = lambda *_: [make_chunk("Already indexed")]
        rag.apipeline_enqueue_documents = AsyncMock()
        rag.apipeline_process_enqueue_documents = AsyncMock(return_value={})

        with patch("minirag.minirag.extract_entities", new_callable=AsyncMock) as extract:
            await rag.ainsert("Already indexed")

        extract.assert_not_awaited()
        rag._insert_done.assert_awaited_once()

    async def test_successful_extraction_marks_only_staged_document_processed(self):
        status_doc = make_status_doc()
        doc_status = FakeDocStatusStorage()
        rag = make_bare_rag(doc_status)
        staged_chunks = {"chunk-new": make_chunk()}
        rag.apipeline_enqueue_documents = AsyncMock()
        rag.apipeline_process_enqueue_documents = AsyncMock(
            return_value={"doc-new": (status_doc, staged_chunks)}
        )

        with (
            patch("minirag.minirag.asdict", return_value={}),
            patch("minirag.minirag.extract_entities", new_callable=AsyncMock) as extract,
        ):
            await rag.ainsert("A document")

        extract.assert_awaited_once()
        self.assertEqual(doc_status.upserts[-1]["doc-new"]["status"], DocStatus.PROCESSED)

    async def test_failed_extraction_marks_document_failed(self):
        status_doc = make_status_doc()
        doc_status = FakeDocStatusStorage()
        rag = make_bare_rag(doc_status)
        staged_chunks = {"chunk-new": make_chunk()}
        rag.apipeline_enqueue_documents = AsyncMock()
        rag.apipeline_process_enqueue_documents = AsyncMock(
            return_value={"doc-new": (status_doc, staged_chunks)}
        )

        with (
            patch("minirag.minirag.asdict", return_value={}),
            patch(
                "minirag.minirag.extract_entities",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "doc-new"),
        ):
            await rag.ainsert("A document")

        self.assertEqual(doc_status.upserts[-1]["doc-new"]["status"], DocStatus.FAILED)
        self.assertEqual(doc_status.upserts[-1]["doc-new"]["error"], "LLM unavailable")
        rag._insert_done.assert_awaited_once()

    async def test_staging_keeps_document_processing_until_extraction(self):
        status_doc = make_status_doc()
        doc_status = FakeDocStatusStorage({DocStatus.PENDING: {"doc-new": status_doc}})
        rag = make_bare_rag(doc_status)
        rag.chunking_func = lambda *_: [make_chunk()]

        staged = await rag.apipeline_process_enqueue_documents()

        self.assertIn("doc-new", staged)
        self.assertEqual(doc_status.upserts[0]["doc-new"]["status"], DocStatus.PROCESSING)
        self.assertEqual(len(rag.chunks_vdb.upserts), 1)


if __name__ == "__main__":
    unittest.main()
