import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
    GraphDatabase,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from minirag.utils import logger
from ..base import BaseGraphStorage
import copy
from minirag.utils import merge_tuples

@dataclass
class MemgraphStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with Memgraph in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        
        # Memgraph connection parameters
        URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
        USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
        PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")
    
        
        # Memgraph doesn't use databases in the same way as Neo4j
        # It uses a single database but can support multiple graphs through labeling
        self._DATABASE = None
        
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI, auth=(USERNAME, PASSWORD) if USERNAME else None
        )
        
        # Test connection
        with GraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD) if USERNAME else None,
        ) as _sync_driver:
            try:
                with _sync_driver.session() as session:
                    try:
                        session.run("MATCH (n) RETURN n LIMIT 0")
                        logger.info(f"Connected to Memgraph at {URI}")
                    except neo4jExceptions.ServiceUnavailable as e:
                        logger.error(f"Memgraph at {URI} is not available")
                        raise e
            except neo4jExceptions.AuthError as e:
                logger.error(f"Authentication failed for Memgraph at {URI}")
                raise e

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def get_types(self) -> tuple[list[str], list[str]]:
        """
        Get all entity types and relationship types from the database
        Returns:
            tuple: (list of node types, list of relationship types)
        """
        node_types = set()
        relationship_types = set()

        async with self._driver.session() as session:
            # Get all node labels (entity types)
            node_query = """
                MATCH (n)
                WITH DISTINCT labels(n) AS node_labels
                UNWIND node_labels AS label
                RETURN DISTINCT label
            """
            
            # Get all relationship types
            rel_query = """
                MATCH ()-[r]->()
                RETURN DISTINCT type(r) AS rel_type
            """
            
            # Execute node types query
            result = await session.run(node_query)
            async for record in result:
                node_types.add(record["label"])
            
            # Execute relationship types query
            result = await session.run(rel_query)
            async for record in result:
                relationship_types.add(record["rel_type"])

        return list(node_types), list(relationship_types)

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id.strip('"')

        async with self._driver.session() as session:
            query = (
                f"MATCH (n:`{entity_name_label}`) RETURN count(n) > 0 AS node_exists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["node_exists"]}'
            )
            return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')

        async with self._driver.session() as session:
            query = (
                f"MATCH (a:`{entity_name_label_source}`)-[r]-(b:`{entity_name_label_target}`) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["edgeExists"]}'
            )
            return single_result["edgeExists"]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self._driver.session() as session:
            entity_name_label = node_id.strip('"')
            query = f"MATCH (n:`{entity_name_label}`) RETURN n"
            result = await session.run(query)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None

    async def get_node_from_types(self, type_list) -> Union[dict, None]:
        node_list = []
        for name, arrt in self._graph.nodes(data=True):
            node_type = arrt.get('entity_type').strip('\"')
            if node_type in type_list:
                node_list.append(name)
        node_datas = await asyncio.gather(
            *[self.get_node(name) for name in node_list]
        )
        node_datas = [
            {**n, "entity_name": k}
            for k, n in zip(node_list, node_datas)
            if n is not None
        ]
        return node_datas

    async def get_neighbors_within_k_hops(self, source_node_id: str, k):
        count = 0
        if await self.has_node(source_node_id):
            source_edge = list(self._graph.edges(source_node_id))
        else:
            print("NO THIS ID:", source_node_id)
            return []
        count = count + 1
        while count < k:
            count = count + 1
            sc_edge = copy.deepcopy(source_edge)
            source_edge = []
            for pair in sc_edge:
                append_edge = list(self._graph.edges(pair[-1]))
                for tuples in merge_tuples([pair], append_edge):
                    source_edge.append(tuples)
        return source_edge

    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id.strip('"')

        async with self._driver.session() as session:
            query = f"""
                MATCH (n:`{entity_name_label}`)
                RETURN SIZE([(n)--() | 1]) AS totalEdgeCount
            """
            result = await session.run(query)
            record = await result.single()
            if record:
                edge_count = record["totalEdgeCount"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                )
                return edge_count
            else:
                return None

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name_label_source = src_id.strip('"')
        entity_name_label_target = tgt_id.strip('"')
        src_degree = await self.node_degree(entity_name_label_source)
        trg_degree = await self.node_degree(entity_name_label_target)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        
        async with self._driver.session() as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}`)-[r]->(end:`{entity_name_label_target}`)
            RETURN properties(r) as edge_properties
            LIMIT 1
            """

            result = await session.run(query)
            record = await result.single()
            if record:
                result = dict(record["edge_properties"])
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}"
                )
                return result
            else:
                return None

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        node_label = source_node_id.strip('"')

        query = f"""MATCH (n:`{node_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected"""
        async with self._driver.session() as session:
            results = await session.run(query)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))

            return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Memgraph database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = node_id.strip('"')
        properties = node_data

        async def _do_upsert(tx: AsyncManagedTransaction):
            query = f"""
            MERGE (n:`{label}`)
            SET n += $properties
            """
            await tx.run(query, properties=properties)
            logger.debug(
                f"Upserted node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
            )
        ),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_label = source_node_id.strip('"')
        target_node_label = target_node_id.strip('"')
        edge_properties = edge_data

        async def _do_upsert_edge(tx: AsyncManagedTransaction):
            query = f"""
            MATCH (source:`{source_node_label}`)
            WITH source
            MATCH (target:`{target_node_label}`)
            MERGE (source)-[r:DIRECTED]->(target)
            SET r += $properties
            RETURN r
            """
            await tx.run(query, properties=edge_properties)
            logger.debug(
                f"Upserted edge from '{source_node_label}' to '{target_node_label}' with properties: {edge_properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Get complete connected subgraph for specified node (including the starting node itself)
        
        Note: Memgraph doesn't have APOC procedures by default, so we use a simpler approach
        """
        label = node_label.strip('"')
        result = {"nodes": [], "edges": []}
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session() as session:
            try:
                # First verify if starting node exists
                validate_query = f"MATCH (n:`{label}`) RETURN n LIMIT 1"
                validate_result = await session.run(validate_query)
                if not await validate_result.single():
                    logger.warning(f"Starting node {label} does not exist!")
                    return result

                # Use a simpler subgraph query without APOC
                # This gets all nodes and relationships within max_depth hops
                main_query = f"""
                MATCH path = (start:`{label}`)-[*0..{max_depth}]-(connected)
                WITH nodes(path) as path_nodes, relationships(path) as path_rels
                UNWIND path_nodes as node
                WITH COLLECT(DISTINCT node) as all_nodes, path_rels
                UNWIND path_rels as rel
                WITH all_nodes, COLLECT(DISTINCT rel) as all_rels
                RETURN all_nodes as nodes, all_rels as relationships
                """
                result_set = await session.run(main_query)
                record = await result_set.single()

                if record:
                    # Handle nodes
                    for node in record["nodes"]:
                        node_id = f"{node.id}_{'_'.join(node.labels)}"
                        if node_id not in seen_nodes:
                            node_data = dict(node)
                            node_data["labels"] = list(node.labels)
                            result["nodes"].append(node_data)
                            seen_nodes.add(node_id)

                    # Handle relationships
                    for rel in record["relationships"]:
                        edge_id = f"{rel.id}_{rel.type}"
                        if edge_id not in seen_edges:
                            edge_data = dict(rel)
                            edge_data.update(
                                {
                                    "source": f"{rel.start_node.id}_{'_'.join(rel.start_node.labels)}",
                                    "target": f"{rel.end_node.id}_{'_'.join(rel.end_node.labels)}",
                                    "type": rel.type,
                                }
                            )
                            result["edges"].append(edge_data)
                            seen_edges.add(edge_id)

                    logger.info(
                        f"Subgraph query successful | Node count: {len(result['nodes'])} | Edge count: {len(result['edges'])}"
                    )

            except neo4jExceptions.ClientError as e:
                logger.error(f"Memgraph query failed: {str(e)}")
                return await self._robust_fallback(label, max_depth)

        return result

    async def _robust_fallback(
        self, label: str, max_depth: int
    ) -> Dict[str, List[Dict]]:
        """Fallback query solution for Memgraph"""
        result = {"nodes": [], "edges": []}
        visited_nodes = set()
        visited_edges = set()

        async def traverse(current_label: str, current_depth: int):
            if current_depth > max_depth:
                return

            # Get current node details
            node = await self.get_node(current_label)
            if not node:
                return

            node_id = f"{current_label}"
            if node_id in visited_nodes:
                return
            visited_nodes.add(node_id)

            # Add node data
            node_data = {k: v for k, v in node.items()}
            node_data["labels"] = [current_label]
            result["nodes"].append(node_data)

            # Get all outgoing and incoming edges
            query = f"""
            MATCH (a)-[r]-(b)
            WHERE a:`{current_label}` OR b:`{current_label}`
            RETURN a, r, b,
                   CASE WHEN startNode(r) = a THEN 'OUTGOING' ELSE 'INCOMING' END AS direction
            """
            async with self._driver.session() as session:
                results = await session.run(query)
                async for record in results:
                    # Handle edges
                    rel = record["r"]
                    edge_id = f"{rel.id}_{rel.type}"
                    if edge_id not in visited_edges:
                        edge_data = dict(rel)
                        edge_data.update(
                            {
                                "source": list(record["a"].labels)[0],
                                "target": list(record["b"].labels)[0],
                                "type": rel.type,
                                "direction": record["direction"],
                            }
                        )
                        result["edges"].append(edge_data)
                        visited_edges.add(edge_id)

                        # Recursively traverse adjacent nodes
                        next_label = (
                            list(record["b"].labels)[0]
                            if record["direction"] == "OUTGOING"
                            else list(record["a"].labels)[0]
                        )
                        await traverse(next_label, current_depth + 1)

        await traverse(label, 0)
        return result

    async def get_all_labels(self) -> List[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        async with self._driver.session() as session:
            # Memgraph compatible query to get all labels
            query = """
                MATCH (n)
                WITH DISTINCT labels(n) AS node_labels
                UNWIND node_labels AS label
                RETURN DISTINCT label
                ORDER BY label
            """

            result = await session.run(query)
            labels = []
            async for record in result:
                labels.append(record["label"])
            return labels