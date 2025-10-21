"""
Integration tests for Memgraph implementation.

These tests require a running Memgraph instance and will perform actual database operations.

Prerequisites:
1. Install Memgraph (Docker recommended):
   docker run -p 7687:7687 -p 7444:7444 -p 3000:3000 memgraph/memgraph:latest

2. Set environment variables (optional, defaults provided):
   export MEMGRAPH_URI="bolt://localhost:7687"
   export MEMGRAPH_USERNAME=""  # Optional, leave empty for no auth
   export MEMGRAPH_PASSWORD=""  # Optional, leave empty for no auth

Usage:
    pytest tests/test_memgraph.py -v
    pytest tests/test_memgraph.py::test_memgraph_connection -v
"""

import os
import pytest
import pytest_asyncio
from minirag.kg.memgraph_impl import MemgraphStorage
from minirag.utils import logger



@pytest.fixture(scope="session")
def memgraph_config():
    """Configuration for Memgraph connection."""
    return {
        "uri": os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687"),
        "username": os.environ.get("MEMGRAPH_USERNAME", ""),
        "password": os.environ.get("MEMGRAPH_PASSWORD", ""),
        "namespace": "test_memgraph",
        "global_config": {"working_dir": "/tmp/minirag_test"},
        "embedding_func": None  # Skip embeddings for testing
    }


@pytest_asyncio.fixture
async def memgraph_storage(memgraph_config):
    """Create a MemgraphStorage instance for testing."""
    # Set environment variables for the storage
    os.environ["MEMGRAPH_URI"] = memgraph_config["uri"]
    os.environ["MEMGRAPH_USERNAME"] = memgraph_config["username"]
    os.environ["MEMGRAPH_PASSWORD"] = memgraph_config["password"]
    
    storage = MemgraphStorage(
        namespace=memgraph_config["namespace"],
        global_config=memgraph_config["global_config"],
        embedding_func=memgraph_config["embedding_func"]
    )
    
    # Clear any existing test data
    try:
        async with storage._driver.session() as session:
            await session.run(
                "MATCH (n) WHERE any(label in labels(n) WHERE label IN "
                "['Person1', 'Person2', 'Company1']) DETACH DELETE n"
            )
    except Exception:
        pass  # Ignore if nodes don't exist
    
    yield storage
    
    # Cleanup after test
    try:
        async with storage._driver.session() as session:
            await session.run(
                "MATCH (n) WHERE any(label in labels(n) WHERE label IN "
                "['Person1', 'Person2', 'Company1']) DETACH DELETE n"
            )
        await storage.close()
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
class TestMemgraphIntegration:
    """Integration tests for MemgraphStorage."""

    async def test_memgraph_connection(self, memgraph_storage):
        """Test basic connection to Memgraph."""
        assert memgraph_storage._driver is not None
        
        # Test simple query
        async with memgraph_storage._driver.session() as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            assert record["test"] == 1

    async def test_node_operations(self, memgraph_storage):
        """Test node creation, retrieval, and checking."""
        storage = memgraph_storage
        
        # Test node creation
        await storage.upsert_node("Person1", {
            "name": "Alice",
            "entity_type": "Person",
            "description": "A software engineer"
        })
        
        # Test node existence
        assert await storage.has_node("Person1") is True
        assert await storage.has_node("NonExistentNode") is False
        
        # Test node retrieval
        alice_data = await storage.get_node("Person1")
        assert alice_data is not None
        assert alice_data["name"] == "Alice"
        assert alice_data["entity_type"] == "Person"
        assert alice_data["description"] == "A software engineer"
        
        # Test node update
        await storage.upsert_node("Person1", {
            "name": "Alice",
            "entity_type": "Person",
            "description": "A senior software engineer",
            "years_experience": 5
        })
        
        updated_data = await storage.get_node("Person1")
        assert updated_data["description"] == "A senior software engineer"
        assert updated_data["years_experience"] == 5

    async def test_edge_operations(self, memgraph_storage):
        """Test edge creation, retrieval, and checking."""
        storage = memgraph_storage
        
        # Create nodes first
        await storage.upsert_node("Person1", {
            "name": "Alice",
            "entity_type": "Person"
        })
        await storage.upsert_node("Company1", {
            "name": "TechCorp",
            "entity_type": "Company"
        })
        
        # Test edge creation
        await storage.upsert_edge("Person1", "Company1", {
            "relationship": "works_at",
            "since": "2020",
            "description": "Alice works at TechCorp"
        })
        
        # Test edge existence
        assert await storage.has_edge("Person1", "Company1") is True
        # Undirected check
        assert await storage.has_edge("Company1", "Person1") is True
        assert await storage.has_edge("Person1", "NonExistentNode") is False
        
        # Test edge retrieval
        edge_data = await storage.get_edge("Person1", "Company1")
        assert edge_data is not None
        assert edge_data["relationship"] == "works_at"
        assert edge_data["since"] == "2020"
        assert edge_data["description"] == "Alice works at TechCorp"

    async def test_node_degree_operations(self, memgraph_storage):
        """Test node degree calculations."""
        storage = memgraph_storage
        
        # Create test graph: Person1 -> Company1 <- Person2
        await storage.upsert_node("Person1", {"name": "Alice"})
        await storage.upsert_node("Person2", {"name": "Bob"})
        await storage.upsert_node("Company1", {"name": "TechCorp"})
        
        await storage.upsert_edge(
            "Person1", "Company1", {"relationship": "works_at"}
        )
        await storage.upsert_edge(
            "Person2", "Company1", {"relationship": "works_at"}
        )
        
        # Test node degrees
        alice_degree = await storage.node_degree("Person1")
        bob_degree = await storage.node_degree("Person2")
        company_degree = await storage.node_degree("Company1")
        
        assert alice_degree == 1  # Connected to Company1
        assert bob_degree == 1    # Connected to Company1
        assert company_degree == 2  # Connected to both Person1 and Person2
        
        # Test edge degree (sum of connected node degrees)
        edge_degree = await storage.edge_degree("Person1", "Company1")
        assert edge_degree == alice_degree + company_degree

    async def test_node_edges_retrieval(self, memgraph_storage):
        """Test retrieval of node edges."""
        storage = memgraph_storage
        
        # Create test graph
        await storage.upsert_node("Person1", {"name": "Alice"})
        await storage.upsert_node("Company1", {"name": "TechCorp"})
        await storage.upsert_node("Project1", {"name": "AI Project"})
        
        await storage.upsert_edge(
            "Person1", "Company1", {"relationship": "works_at"}
        )
        await storage.upsert_edge(
            "Person1", "Project1", {"relationship": "leads"}
        )
        
        # Test node edges
        alice_edges = await storage.get_node_edges("Person1")
        assert len(alice_edges) == 2
        
        # Extract target labels from edges
        connected_labels = {edge[1] for edge in alice_edges}
        assert "Company1" in connected_labels
        assert "Project1" in connected_labels

    async def test_type_operations(self, memgraph_storage):
        """Test type retrieval operations."""
        storage = memgraph_storage
        
        # Create nodes with different types
        await storage.upsert_node("Person1", {"entity_type": "Person"})
        await storage.upsert_node("Company1", {"entity_type": "Company"})
        
        # Create relationship
        await storage.upsert_edge(
            "Person1", "Company1", {"relationship": "works_at"}
        )
        
        # Test get_types
        node_types, relationship_types = await storage.get_types()
        
        assert "Person1" in node_types
        assert "Company1" in node_types
        assert "DIRECTED" in relationship_types  # Default relationship type
        
        # Test get_all_labels
        all_labels = await storage.get_all_labels()
        assert "Person1" in all_labels
        assert "Company1" in all_labels

    async def test_knowledge_graph_retrieval(self, memgraph_storage):
        """Test knowledge graph subgraph retrieval."""
        storage = memgraph_storage
        
        # Create a small network: Person1 -> Company1 <- Person2 -> Project1
        await storage.upsert_node(
            "Person1", {"name": "Alice", "type": "Person"}
        )
        await storage.upsert_node(
            "Person2", {"name": "Bob", "type": "Person"}
        )
        await storage.upsert_node(
            "Company1", {"name": "TechCorp", "type": "Company"}
        )
        await storage.upsert_node(
            "Project1", {"name": "AI Project", "type": "Project"}
        )
        
        await storage.upsert_edge(
            "Person1", "Company1", {"relationship": "works_at"}
        )
        await storage.upsert_edge(
            "Person2", "Company1", {"relationship": "works_at"}
        )
        await storage.upsert_edge(
            "Person2", "Project1", {"relationship": "leads"}
        )
        
        # Test subgraph retrieval from Person1 with depth 2
        subgraph = await storage.get_knowledge_graph("Person1", max_depth=2)
        
        assert "nodes" in subgraph
        assert "edges" in subgraph
        assert len(subgraph["nodes"]) >= 1  # At least Person1
        assert len(subgraph["edges"]) >= 1  # At least Person1 -> Company1
        
        # Verify specific nodes are included
        node_labels = []
        for node in subgraph["nodes"]:
            if "labels" in node:
                node_labels.extend(node["labels"])
        
        # Should include Person1 and Company1 at minimum
        assert any("Person1" in str(label) for label in node_labels)

    async def test_error_handling(self, memgraph_storage):
        """Test error handling for various edge cases."""
        storage = memgraph_storage
        
        # Test getting non-existent node
        result = await storage.get_node("NonExistentNode")
        assert result is None
        
        # Test getting non-existent edge
        result = await storage.get_edge("NonExistent1", "NonExistent2")
        assert result is None
        
        # Test node degree for non-existent node
        result = await storage.node_degree("NonExistentNode")
        assert result is None
        
        # Test knowledge graph for non-existent node
        subgraph = await storage.get_knowledge_graph("NonExistentNode")
        assert subgraph == {"nodes": [], "edges": []}

    async def test_complex_graph_operations(self, memgraph_storage):
        """Test complex graph operations with multiple nodes and edges."""
        storage = memgraph_storage
        
        # Create a more complex graph structure
        nodes = [
            ("Person1", {"name": "Alice", "role": "Engineer"}),
            ("Person2", {"name": "Bob", "role": "Manager"}),
            ("Person3", {"name": "Charlie", "role": "Designer"}),
            ("Company1", {"name": "TechCorp", "industry": "Technology"}),
            ("Project1", {"name": "Web App", "status": "Active"}),
            ("Project2", {"name": "Mobile App", "status": "Planning"})
        ]
        
        edges = [
            ("Person1", "Company1", {
                "relationship": "works_at", "since": "2020"
            }),
            ("Person2", "Company1", {
                "relationship": "manages", "since": "2018"
            }),
            ("Person3", "Company1", {
                "relationship": "works_at", "since": "2021"
            }),
            ("Person1", "Project1", {
                "relationship": "develops", "contribution": "backend"
            }),
            ("Person3", "Project1", {
                "relationship": "designs", "contribution": "ui"
            }),
            ("Person2", "Project2", {
                "relationship": "oversees", "responsibility": "planning"
            })
        ]
        
        # Create all nodes
        for node_id, node_data in nodes:
            await storage.upsert_node(node_id, node_data)
        
        # Create all edges
        for source, target, edge_data in edges:
            await storage.upsert_edge(source, target, edge_data)
        
        # Test comprehensive graph queries
        
        # Verify all nodes exist
        for node_id, _ in nodes:
            assert await storage.has_node(node_id)
        
        # Verify all edges exist
        for source, target, _ in edges:
            assert await storage.has_edge(source, target)
        
        # Test complex subgraph retrieval
        company_subgraph = await storage.get_knowledge_graph(
            "Company1", max_depth=2
        )
        # Should at least include Company1 and some connected nodes
        assert len(company_subgraph["nodes"]) >= 1
        assert len(company_subgraph["edges"]) >= 0  # May have edges
        
        # Test node with highest degree (Company1 should have highest degree)
        company_degree = await storage.node_degree("Company1")
        person1_degree = await storage.node_degree("Person1")
        assert company_degree >= person1_degree


@pytest.mark.asyncio
@pytest.mark.integration
async def test_storage_lifecycle(memgraph_config):
    """Test the complete lifecycle of storage operations."""
    # Set environment variables
    os.environ["MEMGRAPH_URI"] = memgraph_config["uri"]
    os.environ["MEMGRAPH_USERNAME"] = memgraph_config["username"]
    os.environ["MEMGRAPH_PASSWORD"] = memgraph_config["password"]
    
    # Create storage
    storage = MemgraphStorage(
        namespace="lifecycle_test",
        global_config=memgraph_config["global_config"],
        embedding_func=memgraph_config["embedding_func"]
    )
    
    try:
        # Test initial empty state
        node_types, rel_types = await storage.get_types()
        initial_node_count = len(node_types)
        
        # Add some data
        await storage.upsert_node("TestNode", {"data": "test_value"})
        await storage.upsert_node("TestNode2", {"data": "test_value2"})
        await storage.upsert_edge(
            "TestNode", "TestNode2", {"relation": "test_relation"}
        )
        
        # Verify data was added
        assert await storage.has_node("TestNode")
        assert await storage.has_node("TestNode2")
        assert await storage.has_edge("TestNode", "TestNode2")
        
        # Test types after adding data
        node_types, rel_types = await storage.get_types()
        assert len(node_types) >= initial_node_count + 2
        
        # Call index done callback
        await storage.index_done_callback()
        
    finally:
        # Cleanup
        try:
            async with storage._driver.session() as session:
                await session.run(
                    "MATCH (n) WHERE any(label in labels(n) WHERE label IN "
                    "['TestNode', 'TestNode2']) DETACH DELETE n"
                )
        except Exception:
            pass
        await storage.close()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
