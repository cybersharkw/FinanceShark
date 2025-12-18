import os 
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import requests 

#Langchain
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

#kraken
import time
import hmac
import hashlib
import base64

load_dotenv()

router = APIRouter()

checkpoint = InMemorySaver()

BASE_URL = "https://demo-futures.kraken.com"
ENDPOINT = "/derivatives/api/v3/accounts"

API_KEY = os.getenv("KRAKEN_PUBLIC")
API_SECRET = os.getenv("KRAKEN_PRIVATE")


def sign_request(endpoint, nonce, data=""):
    """Create HMAC signature for Kraken API request"""
    message = endpoint + nonce + data
    signature = hmac.new(
        base64.b64decode(API_SECRET),
        message.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

     
def loadPDF():
    """Load PDF document"""
    print("pdf-loaded")    
        
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
def buyCoin(query: str) -> str:
    """Buy cryptocurrency. Use when user wants to purchase crypto."""
    return "Buy feature coming soon"

@tool
def sellCoin(query: str) -> str:
    """Sell cryptocurrency. Use when user wants to sell crypto."""
    return "Sell feature coming soon"

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


def saveHisory(query: str) -> str:
    """Save conversation history"""
    print("test")

# Request model
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    response: str
    session_id: str
    status: str

@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    
    Mymodel = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.0,
    )
    
    agent = create_agent(
        model=Mymodel,
        tools=[newsSummary, PortfolioSummary],   
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