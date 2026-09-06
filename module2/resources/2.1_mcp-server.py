from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from requests import get

load_dotenv()

mcp=FastMCP('mcr_server')
tavily_client = TavilyClient()
@mcp.tool()
def search_web(query: str) -> dict[str, Any]:
    """Search the web using Tavily"""
    return tavily_client.search(query)

@mcp.resource("https://github.com/eresha9992/rating-service/blob/main/README.md")
def github_file():
    """
    Resource acces for langchain
    """
    url="https://github.com/eresha9992/rating-service/blob/main/README.md"
    try:
        resp=get(url)
        return resp.text
    except:
        return f"error:{str(exit)}"

@mcp.prompt()
def prompt():
    """
    Analyze the data from langchain ai repo file with comprehensive insights"
    """
    return """
    You are the helpfull assistant that answers user questions about langchain,Labgraph and Langsmith
    
    you can use the following tools/resources to answer user questions:
    - search_web:search the web information
    - github_file:access the langchain-ai repo files
    
    If the users ask question that is not related to langchain or langsmith you should say Iam sorry
    
    you may be try multiple tool and resources call to answer user questions
    
    you may also ask clarifying questions to the user to better understand their question
    """

if __name__ == "__main__":
    mcp.run(transport="streamable-http")