"""
Items MCP Server - Model Context Protocol server exposing product retrieval as tools.

Learning: MCP (Model Context Protocol) standardizes how AI agents discover and invoke tools
across different servers. FastMCP provides a Python framework to expose functions as MCP tools.
This server runs over HTTP (port 8000 in container, 8001 on host) for production-style deployment.
"""
from fastmcp import FastMCP

from items_mcp_server.core.config import config
from items_mcp_server.utils import retrieve_items_data, process_items_context

# FastMCP instance: name appears in tool metadata; HTTP transport chosen at run()
mcp = FastMCP("items_mcp_server")


@mcp.tool()
def get_formatted_items_context(query: str, top_k: int = 5) -> str:
    """
    Tool invoked by agent when it needs product context. Returns formatted string
    of top-k products (ID, rating, description). Agent uses this to answer questions.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    context = retrieve_items_data(query, k=top_k)
    formatted_context = process_items_context(context)
    return formatted_context


if __name__ == "__main__":
    # HTTP transport: agents connect via Client("http://host:port/mcp"); 0.0.0.0 binds all interfaces
    mcp.run(transport="http", host="0.0.0.0", port=8000)
