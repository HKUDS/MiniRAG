#!/usr/bin/env python3

import asyncio
import os
from dotenv import load_dotenv
from minirag import MiniRAG, QueryParam
from minirag.llm.openai import openai_embed, openai_complete

# Load environment variables
load_dotenv()

async def debug_test():
    print("🔍 Debug test for Memgraph")
    
    # Set up environment
    os.environ.setdefault("MEMGRAPH_URI", "bolt://localhost:7687")
    os.environ.setdefault("MEMGRAPH_USERNAME", "")
    os.environ.setdefault("MEMGRAPH_PASSWORD", "")
    os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    try:
        working_dir = "./debug_minirag"
        os.makedirs(working_dir, exist_ok=True)
        
        rag = MiniRAG(
            working_dir=working_dir,
            graph_storage="MemgraphStorage",
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            embedding_func=openai_embed,
            llm_model_func=openai_complete,
            llm_model_name="gpt-4o-mini",
            log_level="DEBUG"
        )
        
        print("✅ MiniRAG initialized")
        
        # Simple test
        await rag.ainsert("Alice is a software engineer.")
        print("✅ Document inserted")
        
        # Try a query
        result = await rag.aquery("Who is Alice?", param=QueryParam(mode="mini"))
        print(f"💬 Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(debug_test())
    sys.exit(exit_code)
