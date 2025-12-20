import os 
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests 
from pathlib import Path

#Langchain
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
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
from sentence_transformers import SentenceTransformer


#Path for Assets
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

#loads the pdf in assets
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
        
        raise HTTPException(
            status_code=400,
            detail=f"Error loading FOMC PDF: {repr(e)}"
        )
load_dotenv()

router = APIRouter()

checkpoint = MemorySaver()

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
    
        stats = json_data["data"]["quote"]["USD"]
        
        total_mcap = stats.get("total_market_cap", "N/A")
        volume_24h = stats.get("total_volume_24h", "N/A")
        

        result = (
            f"Global Market Cap: ${total_mcap:,.0f} | "
            f"24h Volume: ${volume_24h:,.0f} | "
            f"{fear_greed_report}"
        )
        
        # Initialize embedding model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate embedding for the single result string
        _embedding = _model.encode(result)
        
        # Add to ChromaDB - single document
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        krypto_collection.add(
            ids=[f"coin_metrics_{timestamp}"],  
            documents=[result],  
            embeddings=[_embedding.tolist()], 
            metadatas=[{
                "document": "coin_metrics",
                "timestamp": timestamp,
                "type": "market_data"
            }]
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

import requests
from typing import Dict

@tool
def markedPrice(query: str) -> str:
    """Get current market price for a cryptocurrency. Use when user asks about prices.
    
    Args:
        query: The cryptocurrency name or symbol (e.g., 'bitcoin', 'BTC', 'ethereum', 'ETH')
    
    Returns:
        Current price in USD with additional market data
    """
    
    # Map common names/symbols to Kraken FUTURES symbols
    CRYPTO_MAP = {
        'bitcoin': 'PI_XBTUSD',
        'btc': 'PI_XBTUSD',
        'ethereum': 'PI_ETHUSD',
        'eth': 'PI_ETHUSD',
        'solana': 'PI_SOLUSD',
        'sol': 'PI_SOLUSD',
        'cardano': 'PI_ADAUSD',
        'ada': 'PI_ADAUSD',
        'xrp': 'PI_XRPUSD',
        'ripple': 'PI_XRPUSD',
        'dogecoin': 'PI_DOGEUSD',
        'doge': 'PI_DOGEUSD',
        'polkadot': 'PI_DOTUSD',
        'dot': 'PI_DOTUSD',
        'avalanche': 'PI_AVAXUSD',
        'avax': 'PI_AVAXUSD',
        'polygon': 'PI_MATICUSD',
        'matic': 'PI_MATICUSD',
        'litecoin': 'PI_LTCUSD',
        'ltc': 'PI_LTCUSD',
    }
    
    # Normalize query
    crypto_query = query.lower().strip()
    
    # Get Kraken pair symbol
    pair = CRYPTO_MAP.get(crypto_query)
    
    if not pair:
        available = ', '.join(sorted(set(CRYPTO_MAP.keys())))
        return f"❌ Cryptocurrency '{query}' not found. Available: {available}"
    
    try:
        # Kraken Futures public API endpoint (no API key needed)
        url = f"https://futures.kraken.com/derivatives/api/v3/tickers"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if not data.get('result') or data.get('result') != 'success':
            return f"❌ Kraken API Error: {data.get('error', 'Unknown error')}"
        
        # Extract tickers array
        tickers = data.get('tickers', [])
        
        if not tickers:
            return f"❌ No ticker data available"
        
        # Find the specific ticker for our pair
        ticker = None
        for t in tickers:
            if t.get('symbol') == pair:
                ticker = t
                break
        
        if not ticker:
            return f"❌ No data found for {query} (symbol: {pair})"
        
        # Extract price data from Futures API format
        # Futures API structure is different from Spot API
        current_price = float(ticker.get('last', 0))
        mark_price = float(ticker.get('markPrice', 0))
        change_24h = float(ticker.get('change24h', 0))
        volume_24h = float(ticker.get('vol24h', 0))
        open_24h = float(ticker.get('open24h', 0))
        
        # Calculate high/low from available data
        # Futures API doesn't always provide explicit high/low
        high_24h = current_price * 1.02  # Approximate
        low_24h = current_price * 0.98   # Approximate
        
        change_emoji = "📈" if change_24h >= 0 else "📉"
        
        # Format result
        result_text = (
            f"💰 {query.upper()} Market Data (Kraken Futures)\n"
            f"{'='*40}\n"
            f"Last Price: ${current_price:,.2f}\n"
            f"Mark Price: ${mark_price:,.2f}\n"
            f"24h Change: {change_emoji} {change_24h:+.2f}%\n"
            f"24h Volume: {volume_24h:,.2f}\n"
            f"24h Open: ${open_24h:,.2f}"
        )
        
        # Initialize embedding model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate embedding for the single result string
        _embedding = _model.encode(result_text)
        
        # Add to ChromaDB - single document
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        krypto_collection.add(
            ids=[f"coin_price_{timestamp}"],  
            documents=[result_text],  
            embeddings=[_embedding.tolist()], 
            metadatas=[{
                "document": "coin_price",
                "timestamp": timestamp,
                "type": "market_data"
            }]
        )
        return result_text
        
    except requests.exceptions.Timeout:
        return "❌ Request timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return f"❌ Network Error: {str(e)}"
    except (KeyError, IndexError, ValueError) as e:
        return f"❌ Data parsing error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"



from datetime import datetime

@tool
def recommendation(query: str = "") -> str:
    """
    Analyzes market data and provides investment recommendations.
    
    Retrieves current market news, FOMC statements, and cryptocurrency data,
    then recommends whether to Buy, Sell, or Hold based on analysis.
    
    Args:
        query: Optional specific query for targeted analysis
        
    Returns:
        String containing market analysis and recommendation (Buy/Sell/Hold)
    """
    current_datetime = datetime.now()
    
    try:
        # Retrieve market data
        market_news = finance_news_collection.query(
            f"All information from this week (today: {current_datetime.strftime('%Y-%m-%d')})"
        )
        
        fomc_news = fomc_news_collection.query(
            "latest document from federal reserve"
        )
        
        crypto_news = krypto_collection.query(
            f"latest information about cryptocurrency market and Fear and Greed index "
            f"(today: {current_datetime.strftime('%Y-%m-%d')})"
        )
        
        # Combine all data
        result = {
            "market_news": market_news,
            "fomc_news": fomc_news,
            "crypto_news": crypto_news,
            "timestamp": current_datetime.isoformat()
        }
        
        # TODO: Add analysis logic here to generate actual recommendations
        # This should analyze the retrieved data and return Buy/Sell/Hold
        
        return result
        
    except ConnectionError as e:
        return f"Connection error: Unable to retrieve market data - {str(e)}"
    except ValueError as e:
        return f"Data error: Invalid data format - {str(e)}"
    except Exception as e:
        return f"Error retrieving market data: {str(e)}"


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

        # Initialize embedding model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate embedding for the single result string
        summary_text = "\n".join(summary_lines)
        _embedding = _model.encode(summary_text)
        
        # Add to ChromaDB - single document
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        user_portfolio_collection.add(
            ids=[f"Portfolio_{timestamp}"],  
            documents=[summary_text]  , 
            embeddings=[_embedding.tolist()],  
            metadatas=[{
                "document": "user_portfolio",
                "timestamp": timestamp,
                "type": "portfolio"
            }]
        )
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
        tools=[newsSummary, PortfolioSummary, get_latest_FOMC, CoinMetrics, recommendation, markedPrice],   
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