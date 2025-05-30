import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from weaviate.exceptions import WeaviateQueryException
from minirag.kg.weaviate_impl import (
    run_sync,
    WeaviateVectorStorage,
    WeaviateKVStorage,
    WeaviateGraphStorage,
)

@pytest.mark.asyncio
async def test_run_sync():
    def sync_function(x):
        return x * 2
    result = await run_sync(sync_function, 5)
    assert result == 10

def mock_weaviate_client():
    client = MagicMock()
    client.schema.contains.return_value = False
    client.schema.create_class.return_value = None
    client.data_object.create = MagicMock()
    client.data_object.delete = MagicMock()
    client.data_object.get = MagicMock(return_value={"properties": {"foo": "bar"}})
    client.data_object.exists = MagicMock(return_value=True)
    client.data_object.add_reference = MagicMock()

    # Vector storage
    client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
        "data": {"Get": {"Document": [{"content": "hello"}]}}
    }

    # Count
    client.query.aggregate.return_value.with_meta_count.return_value.do.return_value = {
        "data": {"Aggregate": {"Document": [{"meta": {"count": 2}}]}}
    }

    client.batch.delete_objects = MagicMock()
    client.schema.get.return_value = {"classes": [{"class": "GraphNode"}]}

    return client

# === VECTOR STORAGE ===
@pytest.mark.asyncio
@patch("minirag.kg.weaviate_impl.weaviate.Client")
async def test_vector_storage_methods(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    dummy_global_config = MagicMock()
    dummy_embedding_func = MagicMock()

    # Mock schema check
    mock_client.schema.contains.return_value = False
    mock_client.schema.create_class.return_value = None

    # Setup query chain mock: get().with_near_text().with_limit().do()
    mock_do = MagicMock()
    mock_do.return_value = {
        "data": {"Get": {"Document": [{"content": "hello"}]}}
    }
    mock_with_limit = MagicMock()
    mock_with_limit.do = mock_do
    mock_with_near_text = MagicMock()
    mock_with_near_text.with_limit.return_value = mock_with_limit
    mock_get = MagicMock()
    mock_get.with_near_text.return_value = mock_with_near_text
    mock_client.query.get.return_value = mock_get

    # Mock aggregate().with_meta_count().do()
    mock_agg_do = MagicMock()
    mock_agg_do.do.return_value = {
        "data": {
            "Aggregate": {
                "Document": [
                    {"meta": {"count": 2}}
                ]
            }
        }
    }

    mock_agg = MagicMock()
    mock_agg.with_meta_count.return_value = mock_agg_do
    mock_client.query.aggregate.return_value = mock_agg

    vec = WeaviateVectorStorage(
        namespace="dummy", 
        global_config=dummy_global_config, 
        embedding_func=dummy_embedding_func
    )

    # Test upsert
    await vec.upsert({"doc1": {"content": "hello"}})
    mock_client.data_object.create.assert_called_with({"content": "hello"}, "Document", "doc1")

    # Test query
    results = await vec.query("hello", top_k=1)
    assert results == [{"content": "hello"}]

    # Test delete
    await vec.delete(["doc1"])
    mock_client.data_object.delete.assert_called_with("doc1", "Document")

    # Test clear
    await vec.clear()
    mock_client.batch.delete_objects.assert_called_with(class_name="Document")

    # Test count
    count = await vec.count()
    assert count == 2

    # Test schema init
    mock_client.schema.contains.assert_called_with({"class": "Document"})
    mock_client.schema.create_class.assert_called()

# === KV STORAGE ===

@pytest.mark.asyncio
@patch("minirag.kg.weaviate_impl.weaviate.Client")
async def test_kv_storage_all_methods(mock_client_class):
    mock_client = mock_weaviate_client()
    mock_client_class.return_value = mock_client

    dummy_global_config = MagicMock()
    dummy_embedding_func = MagicMock()

    kv = WeaviateKVStorage(namespace="dummy", 
                           global_config=dummy_global_config, 
                           embedding_func=dummy_embedding_func)

    # get_by_id
    result = await kv.get_by_id("key1")
    assert result == {"foo": "bar"}

    # get_by_ids
    results = await kv.get_by_ids(["key1", "key2"], fields={"foo"})
    assert results == [{"foo": "bar"}, {"foo": "bar"}]

    # all_keys
    mock_do = MagicMock()
    mock_do.return_value = {
        "data": {"Get": {"KVDocument": [{"_id": "1"}, {"_id": "2"}]}}
    }

    mock_with_fields = MagicMock()
    mock_with_fields.do = mock_do

    mock_with_additional = MagicMock()
    mock_with_additional.with_fields.return_value = mock_with_fields

    mock_get = MagicMock()
    mock_get.do.return_value = {
        "data": {"Get": {"KVDocument": [{"_id": "1"}, {"_id": "2"}]}}
    }
    mock_client.query.get.return_value = mock_get

    keys = await kv.all_keys()
    assert keys == ["1", "2"]

    # filter_keys
    filtered = await kv.filter_keys(["1", "3"])
    assert filtered == {"3"}

    # upsert
    await kv.upsert({"k1": {"key": "value"}})
    mock_client.data_object.create.assert_called()

    # delete
    await kv.delete(["k1"])
    mock_client.data_object.delete.assert_called()

    # clear
    await kv.clear()
    mock_client.batch.delete_objects.assert_called()

    # count
    mock_count_do = MagicMock(return_value={
        "data": {"Aggregate": {"KVDocument": [{"meta": {"count": 5}}]}}
    })
    mock_with_meta = MagicMock()
    mock_with_meta.do = mock_count_do
    mock_aggregate = MagicMock()
    mock_aggregate.with_meta_count.return_value = mock_with_meta
    mock_client.query.aggregate.return_value = mock_aggregate
    mock_client.schema.contains.return_value = False
    kv.init_schema()
    mock_client.schema.create_class.assert_called_with({
        "class": "KVDocument",
        "properties": [{"name": "key", "dataType": ["text"]}]
    })

    count = await kv.count()
    assert count == 5

# === GRAPH STORAGE ===

@pytest.mark.asyncio
@patch("minirag.kg.weaviate_impl.weaviate.Client")
async def test_graph_storage_all_methods(mock_client_class):
    mock_client = mock_weaviate_client()
    mock_client_class.return_value = mock_client

    dummy_global_config = MagicMock()

    graph = WeaviateGraphStorage(namespace="dummy", 
                                 global_config=dummy_global_config)
        # init_schema
    mock_client.schema.contains.return_value = False
    graph.init_schema()
    mock_client.schema.create_class.assert_called_with({
        "class": "GraphNode",
        "properties": [
            {"name": "name", "dataType": ["text"]},
            {"name": "linkedTo", "dataType": ["GraphNode"]}
        ]
    })


    # get_types
    node_types, edge_types = await graph.get_types()
    assert "GraphNode" in node_types
    assert edge_types == []

    # has_node
    assert await graph.has_node("node1") is True

    # has_edge
    assert await graph.has_edge("a", "b") is False

    # node_degree and edge_degree
    assert await graph.node_degree("n1") == 0
    assert await graph.edge_degree("a", "b") == 0

    # get_node
    result = await graph.get_node("node1")
    assert result == {"foo": "bar"}

    # get_edge and get_node_edges
    assert await graph.get_edge("a", "b") is None
    assert await graph.get_node_edges("a") is None

    # upsert_node
    await graph.upsert_node("nodeX", {"name": "hello"})
    mock_client.data_object.create.assert_called()

    # upsert_edge
    await graph.upsert_edge("nodeX", "nodeY", {})
    mock_client.data_object.add_reference.assert_called()

    # delete_node
    await graph.delete_node("nodeX")
    mock_client.data_object.delete.assert_called()

    # clear
    await graph.clear()
    mock_client.batch.delete_objects.assert_called()

    # embed_nodes
    with pytest.raises(NotImplementedError):
        await graph.embed_nodes("dummy-algo")
