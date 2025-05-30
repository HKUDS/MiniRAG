import asyncio
from typing import Any, Union, List, Set, Dict
import weaviate
from weaviate.exceptions import WeaviateQueryException
from dataclasses import dataclass, field
from minirag.base import (
    BaseVectorStorage,
    BaseKVStorage,
    BaseGraphStorage
)

async def run_sync(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

@dataclass
class WeaviateVectorStorage(BaseVectorStorage):
    client: weaviate.Client = field(init=False)

    def __post_init__(self):
        self.client = weaviate.Client(url="http://localhost:8080")
        self.init_schema()
        # We assume `namespace` holds the weaviate URL
    def init_schema(self):
        try:
            if not self.client.schema.contains({"class": "Document"}):
                self.client.schema.create_class({
                    "class": "Document",
                    "properties": [{"name": "content", "dataType": ["text"]}],
                })
        except WeaviateQueryException as e:
            print(f"Vector schema init error: {e}")

    async def query(self, query: str, top_k: int) -> List[Dict]:
        try:
            near_text = {"concepts": [query]}
            response = await run_sync(
                self.client.query.get, "Document", ["content"]
            )
            response = (
                self.client.query.get("Document", ["content"])
                .with_near_text(near_text)
                .with_limit(top_k)
                .do()
            )
            return response.get("data", {}).get("Get", {}).get("Document", [])
        except WeaviateQueryException as e:
            print(f"Weaviate query error: {e}")
            return []

    async def upsert(self, data: Dict[str, Dict]):
        # data: {id: {content: str, embedding: list[float], ...}}
        for _id, value in data.items():
            try:
                # Weaviate expects data objects for upsert
                obj = {
                    "content": value.get("content"),
                }
                await run_sync(
                    self.client.data_object.create, obj, "Document", _id
                )
            except WeaviateQueryException as e:
                print(f"Error upserting id {_id} to weaviate: {e}")

    async def delete(self, ids: List[str]):
        for _id in ids:
            try:
                await run_sync(self.client.data_object.delete, _id, "Document")
            except WeaviateQueryException as e:
                print(f"Error deleting id {_id}: {e}")

    async def clear(self):
        try:
            await run_sync(self.client.batch.delete_objects, class_name="Document")
        except WeaviateQueryException as e:
            print(f"Vector clear failed: {e}")

    async def count(self) -> int:
        try:
            result = await run_sync(
                self.client.query.aggregate("Document").with_meta_count().do
            )
            return result["data"]["Aggregate"]["Document"][0]["meta"]["count"]
        except WeaviateQueryException as e:
            print(f"Count failed: {e}")
            return 0

@dataclass
class WeaviateKVStorage(BaseKVStorage):
    client: weaviate.Client = field(init=False)

    def __post_init__(self):
        self.client = weaviate.Client(url="http://localhost:8080")
        self.init_schema()

    async def all_keys(self) -> List[str]:
        try:
            response = await run_sync(lambda: self.client.query.get("KVDocument", ["_id"]).do())
            results = response.get("data", {}).get("Get", {}).get("KVDocument", [])
            return [item["_id"] for item in results]
        except WeaviateQueryException as e:
            print(f"Weaviate all_keys error: {e}")
            return []

    async def get_by_id(self, id: str) -> Union[Dict, None]:
        try:
            obj = await run_sync(self.client.data_object.get, id, "KVDocument")
            return obj.get("properties", None)
        except WeaviateQueryException as e:
            print(f"Weaviate get_by_id error for {id}: {e}")
            return None

    async def get_by_ids(self, ids: List[str], fields: Union[Set[str], None] = None) -> List[Union[Dict, None]]:
        results = []
        for _id in ids:
            data = await self.get_by_id(_id)
            if data and fields:
                data = {k: v for k, v in data.items() if k in fields}
            results.append(data)
        return results

    async def filter_keys(self, data: List[str]) -> Set[str]:
        existing_keys = set(await self.all_keys())
        return set(data) - existing_keys

    async def upsert(self, data: Dict[str, Dict]):
        for _id, value in data.items():
            try:
                await run_sync(
                    self.client.data_object.create, value, "KVDocument", _id
                )
            except WeaviateQueryException as e:
                print(f"Error upserting KV id {_id}: {e}")

    async def delete(self, ids: List[str]):
        for _id in ids:
            try:
                await run_sync(self.client.data_object.delete, _id, "KVDocument")
            except WeaviateQueryException as e:
                print(f"Error deleting KV id {_id}: {e}")

    def init_schema(self):
        try:
            if not self.client.schema.contains({"class": "KVDocument"}):
                self.client.schema.create_class({
                    "class": "KVDocument",
                    "properties": [{"name": "key", "dataType": ["text"]}]
                })
        except WeaviateQueryException as e:
            print(f"KV schema init error: {e}")

    async def clear(self):
        try:
            await run_sync(self.client.batch.delete_objects, class_name="KVDocument")
        except WeaviateQueryException as e:
            print(f"KV clear failed: {e}")

    async def count(self) -> int:
        try:
            result = await run_sync(lambda: self.client.query.aggregate("KVDocument").with_meta_count().do())
            return result["data"]["Aggregate"]["KVDocument"][0]["meta"]["count"]
        except WeaviateQueryException as e:
            print(f"KV count failed: {e}")
            return 0
        
@dataclass
class WeaviateGraphStorage(BaseGraphStorage):
    client: weaviate.Client = field(init=False)

    def __post_init__(self):
        self.client = weaviate.Client(url="http://localhost:8080")
        self.init_schema()

    async def get_types(self) -> tuple[List[str], List[str]]:
        # Returns (list of node classes, list of edge classes)
        try:
            schema = await run_sync(self.client.schema.get)
            classes = schema.get("classes", [])
            node_classes = [cls["class"] for cls in classes]
            # Weaviate edges are references inside properties, so edge classes uncommon
            edge_classes = []
            return node_classes, edge_classes
        except WeaviateQueryException as e:
            print(f"Weaviate get_types error: {e}")
            return [], []

    async def has_node(self, node_id: str) -> bool:
        try:
            obj = await run_sync(self.client.data_object.exists, node_id, None)
            return obj
        except WeaviateQueryException as e:
            print(f"Error checking node {node_id}: {e}")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        # Weaviate edges are references; this is complex, simplified here
        return False

    async def node_degree(self, node_id: str) -> int:
        # Not trivial in Weaviate, returning 0 or implement with custom query
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return 0

    async def get_node(self, node_id: str) -> Union[Dict, None]:
        try:
            obj = await run_sync(self.client.data_object.get, node_id)
            return obj.get("properties", None)
        except WeaviateQueryException as e:
            print(f"Error getting node {node_id}: {e}")
            return None

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[Dict, None]:
        return None

    async def get_node_edges(self, source_node_id: str) -> Union[List[tuple], None]:
        return None

    async def upsert_node(self, node_id: str, node_data: Dict[str, str]):
        try:
            await run_sync(self.client.data_object.create, node_data, "GraphNode", node_id)
        except WeaviateQueryException as e:
            print(f"Error upserting node {node_id}: {e}")
    def init_schema(self):
        try:
            if not self.client.schema.contains({"class": "GraphNode"}):
                self.client.schema.create_class({
                    "class": "GraphNode",
                    "properties": [
                        {"name": "name", "dataType": ["text"]},
                        {"name": "linkedTo", "dataType": ["GraphNode"]}  # For references
                    ]
                })
        except WeaviateQueryException as e:
            print(f"Graph schema init error: {e}")

    async def clear(self):
        try:
            await run_sync(self.client.batch.delete_objects, class_name="GraphNode")
        except WeaviateQueryException as e:
            print(f"Graph clear failed: {e}")

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, str]):
        try:
            await run_sync(
                self.client.data_object.add_reference,
                from_object_uuid=source_node_id,
                from_property="linkedTo",
                to_object_uuid=target_node_id
            )
        except WeaviateQueryException as e:
            print(f"Edge creation failed: {e}")

    async def delete_node(self, node_id: str):
        try:
            await run_sync(self.client.data_object.delete, node_id, "GraphNode")
        except WeaviateQueryException as e:
            print(f"Error deleting node {node_id}: {e}")

    async def embed_nodes(self, algorithm: str):
        raise NotImplementedError("Node embedding not implemented for Weaviate.")
