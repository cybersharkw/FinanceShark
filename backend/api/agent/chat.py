import os 
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import requests 
import uuid
from pathlib import Path

#Langchain
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#kraken
import time
import hmac
import hashlib
import base64

#chromaDb
import chromadb
from datetime import datetime
from bs4 import BeautifulSoup
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
class URLRequest(BaseModel):
    url: str
    collection_name: Optional[str] = "finance_news"
# Request model
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    response: str
    session_id: str
    status: str
    
# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./FinanceShark/chroma_db")

# Create collections
responses = chroma_client.get_or_create_collection("chat_responses")

fomc_news_collection = chroma_client.get_or_create_collection(
    name="latest_fomc",
    metadata={"description": "latest report from the fomc meeting"}
)

finance_news_collection = chroma_client.get_or_create_collection(
    name="finance_news",
    metadata={"description": "Stores important finance news agent responses"}
)

krypto_collection = chroma_client.get_or_create_collection(
    name="krypto-data",
    metadata={"description": "Stores important krypto metrics agent responses"}
)

user_portfolio_collection = chroma_client.get_or_create_collection(
    name="User-Portfolio",
    metadata={"description": "Stores responses about the user portfolio"}
)

# Collection mapping for easy access
COLLECTIONS = {
    "chat_responses": responses,
    "fomc": fomc_news_collection,
    "finance_news": finance_news_collection,
    "krypto": krypto_collection,
    "portfolio": user_portfolio_collection
}


def save_to_chromadb(
    collection_name: str, 
    content: List[str],  # Changed from str to List[str]
    embeddings: Optional[List] = None, # Added this
    metadata: Optional[List[Dict]] = None, # Changed from Dict to List[Dict]
    ids: Optional[List[str]] = None # Added this
) -> Dict:
    try:
        collection = COLLECTIONS.get(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' not found.")
        
        # Save to ChromaDB (Chroma's .add() expects lists)
        collection.add(
            documents=content,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        return {"status": "success"}
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def retrieve_from_chromadb(
    collection_name: str,
    query: str,
    n_results: int = 5,
    filter_metadata: Optional[Dict] = None
) -> Dict:
    """
    Universal retrieve function for all ChromaDB collections
    
    Args:
        collection_name: Name of the collection (key from COLLECTIONS dict)
        query: Search query text
        n_results: Number of results to return (default: 5)
        filter_metadata: Optional metadata filter (e.g., {"source": "html"})
    
    Returns:
        Dict with query results including documents, metadata, and distances
    """
    try:
        # Get the collection
        collection = COLLECTIONS.get(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' not found. Available: {list(COLLECTIONS.keys())}")
        
        # Query ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata if filter_metadata else None
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "id": results['ids'][0][i] if results['ids'] else None
                })
        
        print(f"✓ Retrieved {len(formatted_results)} results from {collection_name}")
        
        return {
            "status": "success",
            "collection": collection_name,
            "query": query,
            "count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        print(f"✗ Error retrieving from {collection_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving from ChromaDB: {str(e)}")


def get_collection_info(collection_name: str) -> Dict:
    """Get information about a specific collection"""
    try:
        collection = COLLECTIONS.get(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        count = collection.count()
        return {
            "collection": collection_name,
            "document_count": count,
            "metadata": collection.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")

    
    
def loadFOMC():
    """Load latest FOMC data and save to ChromaDB"""

    pdf_path = BACKEND_DIR / "assets" / "fomc_december_2025.pdf"
    try:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])

        # Chunk PDF
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        _chunks = splitter.split_text(full_text)
        
        # Create embeddings
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _embeddings = _model.encode(_chunks)
        
        fomc_news_collection.add(
        ids=[f"fomc_{i}" for i in range(len(_chunks))],
        documents=_chunks,
        embeddings=_embeddings,
        metadatas=[{"document": "fomc_december_2025.pdf"}] * len(_chunks)
        )
        return full_text
    except Exception as e:
        # Repr(e) helps see the exact error like TypeError
        raise HTTPException(
            status_code=400,
            detail=f"Error loading FOMC PDF: {repr(e)}"
        )
load_dotenv()

router = APIRouter()

checkpoint = InMemorySaver()

BASE_URL = "https://demo-futures.kraken.com"
ENDPOINT = "/derivatives/api/v3/accounts"

API_KEY = os.getenv("KRAKEN_PUBLIC")
API_SECRET = os.getenv("KRAKEN_PRIVATE")
COIN_KEY = os.getenv("COINMARKETCAP_API_KEY")

@tool
def CoinMetrics(query: str = "") -> str:
    """latest data about Kryptocurrencie Market. Including Fear and Gread Index. Which indicates which Season we are currently in"""
    url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
    headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': COIN_KEY,
    }
    
    fear_greed_report = FearGread()
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_data = response.json()
        
        # FIX 2: Correctly navigate the nested JSON for Global Metrics
        # Path: data -> quote -> USD
        stats = json_data["data"]["quote"]["USD"]
        
        total_mcap = stats.get("total_market_cap", "N/A")
        volume_24h = stats.get("total_volume_24h", "N/A")
        
        # FIX 3: Combine strings using f-strings (don't use .join() for single strings)
        result = (
            f"Global Market Cap: ${total_mcap:,.0f} | "
            f"24h Volume: ${volume_24h:,.0f} | "
            f"{fear_greed_report}"
        )
        return result
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"
    

def FearGread():
    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"
    
    headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': COIN_KEY,
    }

    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_data = response.json()
        

        inner_data = json_data.get("data", {})
        value = inner_data.get("value")
        update_time = inner_data.get("update_time")
        
        if value is None:
            return "Error: Could not find 'value' in API response."
            
        return f"Fear & Greed: {value} (Updated: {update_time})"
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"
    
    

def sign_request(endpoint, nonce, data=""):
    """Create HMAC signature for Kraken API request"""
    message = endpoint + nonce + data
    signature = hmac.new(
        base64.b64decode(API_SECRET),
        message.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

@tool
def get_latest_FOMC(query: str = "") -> str:
    """Get the latest FOMC meeting statement from the Federal Reserve. 
    This tool fetches, saves, and returns the most recent FOMC press release."""
    try:
        text = loadFOMC()
        

        return text
        
    except Exception as e:
        return f"Error loading FOMC data: {str(e)}"

@tool
def newsSummary(query: str) -> str:
    """Get the latest cryptocurrency news and market updates. Use this when user asks about crypto news, market trends, or recent developments."""
    api_key = os.getenv("MARKETAU_API_KEY")
    
    url = "https://api.marketaux.com/v1/news/all"
    
    params = {
        "filter_entities": "true",
        "language": "en",
        "api_token": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("data"):
            news_items = []
            for article in data["data"][:3]:  # Top 3 articles
                news_items.append(f"- {article.get('title', 'No title')}: {article.get('description', 'No description')}")
            return "\n".join(news_items)
        else:
            return "No news found"
    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"

@tool
def markedPrice(query: str) -> str:
    """Get current market price for a cryptocurrency. Use when user asks about prices."""
    
    return "Price feature coming soon"

@tool
def recommandation(query: str = "") -> str:
    """You must recommend to the user which action he should take depending on the market situation and previous conversations. Hold the coin, sale or buy. """
    return "Action: Hold. Reason: Market volatility is high."
@tool
def PortfolioSummary(query: str = "") -> str:
    """Get the user's current Kraken demo futures portfolio balance showing all assets and amounts."""
    nonce = str(int(time.time() * 1000))
    signature = sign_request(ENDPOINT, nonce)

    headers = {
        "APIKey": API_KEY,
        "Nonce": nonce,
        "Authent": signature
    }

    try:
        resp = requests.get(BASE_URL + ENDPOINT, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        accounts = data.get("accounts", {})

        summary_lines = ["Demo Futures Portfolio:"]

        # ---- Cash account ----
        cash = accounts.get("cash")
        if cash:
            summary_lines.append("\nCash Account:")
            for currency, balance in cash.get("balances", {}).items():
                summary_lines.append(f"• {currency}: {balance}")

        # ---- Flex (multi-collateral) account ----
        flex = accounts.get("flex")
        if flex:
            summary_lines.append("\nMulti-Collateral Wallet:")
            for currency, info in flex.get("currencies", {}).items():
                qty = info.get("quantity", 0)
                value = info.get("value", 0)
                summary_lines.append(f"• {currency}: {qty} (USD {value})")

        # ---- Margin accounts (dynamic keys) ----
        for name, account in accounts.items():
            if account.get("type") == "marginAccount":
                summary_lines.append(f"\nMargin Account ({name}):")
                for currency, balance in account.get("balances", {}).items():
                    summary_lines.append(f"• {currency}: {balance}")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"Failed to load portfolio: {e}"

@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    
    Mymodel = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.0,
    )
    
    agent = create_agent(
        model=Mymodel,
        tools=[newsSummary, PortfolioSummary, get_latest_FOMC, CoinMetrics, recommandation],   
        checkpointer=checkpoint,
        context_schema="update"
    )
    
    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": "You are Finance-Trader. Your job is to help improve the portfolio of the user with your market insights."},
                {"role": "user", "content": request.query}
            ]
        },
        config={"configurable": {"thread_id": request.session_id}}
    )
    
    ai_message = result["messages"][-1]
    
    return QueryResponse(
        response=ai_message.content,
        session_id=request.session_id,
        status="success"
    )