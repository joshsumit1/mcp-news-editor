import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from supabase import Client, create_client

from mcp import types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Load environment variables from .env file (if present)
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_BASE = get_env_var("API_BASE_URL")  # Your backend on Render
SUPABASE_URL = get_env_var("SUPABASE_URL")
SUPABASE_KEY = get_env_var("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")

PROCESSED_COLUMN = os.getenv("PROCESSED_COLUMN", "professional_rewrite")

# Railway usually provides PORT. Default to 8000 locally.
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Transport:
# - streamable-http => best for Railway / remote deployment
# - stdio => local MCP client usage (Claude Desktop, etc.)
TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http").lower()

# -----------------------------
# Initialize clients
# -----------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# Prompt
# -----------------------------
EDITOR_PROMPT_TEMPLATE = """You are a senior news editor at a major international publication.
Rewrite the following article in a clear, authoritative, and engaging journalistic style.
Follow these rules strictly:
- Inverted pyramid: most important facts first.
- Short paragraphs (2-3 sentences each).
- Remove all promotional or clickbait language.
- Keep all factual claims, names, dates, and numbers unchanged.
- Add a strong, neutral, and informative headline (max 12 words).
- Use active voice and avoid clichés.
- Do NOT add opinions or extra facts.

Output must be valid JSON with exactly two fields: "headline" and "body".

Original article:
Title: {title}
Content: {content}
"""


# -----------------------------
# Helpers
# -----------------------------
async def fetch_articles_to_process(limit: int = 3) -> list[dict[str, Any]]:
    """
    Fetch articles that still have NULL in professional_rewrite column.
    """
    try:
        result = (
            supabase.table("ai_news")
            .select("id, title, ai_content, short_desc")
            .is_(PROCESSED_COLUMN, "null")
            .limit(limit)
            .execute()
        )
        articles = result.data or []
        if articles:
            print(f"📥 Fetched {len(articles)} articles from Supabase")
            return articles
    except Exception as e:
        print(f"❌ Supabase query failed: {e}")

    # Fallback: API endpoint
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{API_BASE}/api/news",
                params={"limit": limit, "no_professional": "true"},
                timeout=10.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("data", []) or []
            print(f"📥 Fetched {len(articles)} articles via API fallback")
            return articles
    except Exception as e:
        print(f"⚠️ API fallback failed: {e}")

    return []


async def rewrite_professionally(title: str, original_content: str) -> dict[str, str]:
    """
    Send prompt to Gemini and parse JSON response.
    """
    prompt = EDITOR_PROMPT_TEMPLATE.format(title=title, content=original_content)

    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.6,
            response_mime_type="application/json",
        ),
    )

    raw_text = (response.text or "").strip()

    # Extract JSON if Gemini wrapped it in markdown
    if "```json" in raw_text:
        raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```", 1)[1].split("```", 1)[0]

    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}. Using fallback.")
        parsed = {"headline": title, "body": raw_text}

    return {
        "headline": str(parsed.get("headline", title)),
        "body": str(parsed.get("body", original_content)),
    }


async def process_article(article: dict[str, Any]) -> None:
    """
    Rewrite one article and update Supabase.
    """
    article_id = article["id"]
    original_title = article.get("title", "")
    base_content = article.get("ai_content") or article.get("short_desc") or ""

    if not base_content:
        print(f"⚠️ Article {article_id} has no content, skipping")
        return

    try:
        rewritten = await rewrite_professionally(original_title, base_content)
        full_text = f"{rewritten['headline']}\n\n{rewritten['body']}"

        supabase.table("ai_news").update(
            {
                PROCESSED_COLUMN: full_text,
                "professional_rewrite_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", article_id).execute()

        print(f"✅ Article {article_id} rewritten professionally")
    except Exception as e:
        print(f"❌ Error rewriting article {article_id}: {e}")


async def fetch_and_process_batch() -> None:
    """
    Main batch function: get up to 3 unprocessed articles, rewrite, update DB.
    """
    articles = await fetch_articles_to_process(limit=3)
    if not articles:
        print(f"[{datetime.now(timezone.utc).isoformat()}] No pending articles.")
        return

    print(f"Processing {len(articles)} article(s)...")
    for art in articles:
        await process_article(art)


# -----------------------------
# MCP Server Setup
# -----------------------------
async def handle_list_tools(ctx, params):
    return mcp_types.ListToolsResult(
        tools=[
            mcp_types.Tool(
                name="process_professional_rewrite",
                description=(
                    "Fetch up to 3 articles without professional rewrite, "
                    "rewrite them with Gemini, and store the result."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )
        ]
    )


async def handle_call_tool(ctx, params):
    if params.name != "process_professional_rewrite":
        raise ValueError(f"Unknown tool: {params.name}")

    await fetch_and_process_batch()
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Batch processed.")]
    )


server = Server(
    "professional-news-editor",
    on_list_tools=handle_list_tools,
    on_call_tool=handle_call_tool,
)

# -----------------------------
# Background scheduler
# -----------------------------
async def background_worker() -> None:
    while True:
        try:
            await fetch_and_process_batch()
        except Exception as e:
            print(f"❌ Background worker error: {e}")
        await asyncio.sleep(60)


def start_background_worker_thread() -> None:
    """
    Run the infinite async background worker in its own daemon thread.
    """
    def runner() -> None:
        try:
            asyncio.run(background_worker())
        except Exception as e:
            print(f"❌ Background worker thread stopped: {e}")

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


# -----------------------------
# Main entry point
# -----------------------------
async def run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_streamable_http() -> None:
    import uvicorn
    app = server.streamable_http_app()
    uvicorn.run(app, host=HOST, port=PORT)


async def main() -> None:
    print("🚀 Starting Professional News Editor MCP Server")
    print(f"Supabase URL: {SUPABASE_URL[:20]}...")
    print(f"Gemini API Key: {'Set' if GEMINI_API_KEY else 'Missing'}")
    print(f"Processing column: {PROCESSED_COLUMN}")
    print(f"Transport: {TRANSPORT}")

    start_background_worker_thread()

    if TRANSPORT == "stdio":
        await run_stdio()
    else:
        run_streamable_http()


if __name__ == "__main__":
    asyncio.run(main())