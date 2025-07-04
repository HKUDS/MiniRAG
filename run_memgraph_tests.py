#!/usr/bin/env python3
"""
Script to run Memgraph integration tests.

This script sets up the environment and runs the Memgraph integration tests.
It can be used to test the Memgraph implementation with a real database.

Prerequisites:
1. Install Memgraph (Docker recommended):
   docker run -d -p 7687:7687 -p 7444:7444 -p 3000:3000 --name memgraph memgraph/memgraph:latest

2. Install dependencies:
   pip install -r requirements.txt

Usage:
    python run_memgraph_tests.py
    python run_memgraph_tests.py --verbose
    python run_memgraph_tests.py --test test_node_operations
"""

import os
import sys
import subprocess
import argparse
import time


def check_memgraph_connection():
    """Check if Memgraph is accessible."""
    try:
        from neo4j import GraphDatabase
        
        uri = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
        username = os.environ.get("MEMGRAPH_USERNAME", "")
        password = os.environ.get("MEMGRAPH_PASSWORD", "")
        
        with GraphDatabase.driver(uri, auth=(username, password) if username else None) as driver:
            with driver.session() as session:
                session.run("RETURN 1")
        
        print("✅ Memgraph connection successful")
        return True
    except Exception as e:
        print(f"❌ Memgraph connection failed: {e}")
        print("\nPlease ensure Memgraph is running:")
        print("docker run -d -p 7687:7687 -p 7444:7444 -p 3000:3000 --name memgraph memgraph/memgraph:latest")
        return False


def run_tests(test_name=None, verbose=False):
    """Run the Memgraph integration tests."""
    cmd = ["python", "-m", "pytest", "tests/test_memgraph.py"]
    
    if test_name:
        cmd.append(f"::TestMemgraphIntegration::{test_name}")
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-m", "integration", "--tb=short"])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Memgraph integration tests")
    parser.add_argument("--test", help="Specific test method to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-connection-check", action="store_true", 
                       help="Skip Memgraph connection check")
    
    args = parser.parse_args()
    
    print("🚀 Memgraph Integration Test Runner")
    print("=" * 50)
    
    # Set default environment variables
    os.environ.setdefault("MEMGRAPH_URI", "bolt://localhost:7687")
    os.environ.setdefault("MEMGRAPH_USERNAME", "")
    os.environ.setdefault("MEMGRAPH_PASSWORD", "")
    
    print(f"Memgraph URI: {os.environ['MEMGRAPH_URI']}")
    
    if not args.skip_connection_check:
        print("\n📡 Checking Memgraph connection...")
        if not check_memgraph_connection():
            sys.exit(1)
    
    print("\n🧪 Running integration tests...")
    try:
        exit_code = run_tests(args.test, args.verbose)
        if exit_code == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
