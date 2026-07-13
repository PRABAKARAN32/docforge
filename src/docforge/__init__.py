"""DocForge — keeps a documentation RAG in sync by updating only what changed.

The top-level package holds only the version; the real work lives in focused modules:
crawling (``crawler``), hashing (``hashing``), the manifest (``manifest``), change
detection (``diff``/``detector``), embedding and the vector store (``embedder``/
``vectorstore``/``rag``), the CLI (``cli``), and the MCP server (``mcp_server``).
"""

__version__ = "0.1.0"
