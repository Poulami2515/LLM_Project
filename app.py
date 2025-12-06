import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from openai import OpenAI

# ---------- Config ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "feedback.json"

OPENAI_MODEL = "gpt-4.1-mini"  # or gpt-4.1, etc.

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Two-Dashboard AI Feedback System")

# CORS so frontends (if separate) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Data Model ----------
class FeedbackItem(BaseModel):
    id: str
    timestamp: str
    rating: int
    review: str
    ai_summary: str
    ai_recommended_actions: str
    ai_user_response: str


def load_data() -> List[dict]:
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(items: List[dict]):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


# ---------- LLM Helpers ----------
''' def generate_ai_analysis(rating: int, review: str):
    """
    Calls the LLM once and asks it to produce:
    - short summary for admin
    - recommended actions for admin
    - user-facing friendly reply
    """
    system_msg = (
        "You are an assistant helping a product team process user feedback.\n"
        "You must ALWAYS respond in strict JSON with keys: "
        "'summary', 'recommended_actions', 'user_response'. "
        "No extra text."
    )

    user_msg = f"""
User rating (1-5): {rating}
User review: {review}

Return:
- 'summary': 1–2 sentence neutral summary of the feedback
- 'recommended_actions': 2–4 bullet-style suggestions for what the team should do next
- 'user_response': a warm, empathetic reply to the user in 2–4 sentences
"""

    completion = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ],
        response_format={"type": "json_object"},
    )

    content = completion.output[0].content[0].text
    data = json.loads(content)

    return (
        data.get("summary", "").strip(),
        data.get("recommended_actions", "").strip(),
        data.get("user_response", "").strip(),
    )'''

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_ai_analysis(rating: int, review: str):
    system_msg = (
        "You are an assistant helping a product team process user feedback. "
        "Respond ONLY in JSON with keys: summary, recommended_actions, user_response."
    )

    user_msg = f"""
User rating: {rating}
Review: {review}

Return:
- 'summary': 1–2 sentence summary
- 'recommended_actions': bullet list
- 'user_response': friendly reply
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )

        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)

        return (
            data.get("summary", ""),
            data.get("recommended_actions", ""),
            data.get("user_response", ""),
        )
    except Exception as e:
        print("LLM Error:", e)
        return (
            "AI summary unavailable due to an error.",
            "Check the logs for LLM error and retry.",
            "Thanks for your feedback! Sorry — our AI couldn't respond properly this time.",
        )


# ---------- Routes: Dashboards ----------
@app.get("/", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    """
    Public-facing user dashboard.
    """
    return templates.TemplateResponse("user.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """
    Internal-facing admin dashboard.
    """
    return templates.TemplateResponse("admin.html", {"request": request})


# ---------- API ----------
@app.post("/api/feedback")
async def create_feedback(
    rating: int = Body(..., embed=True),
    review: str = Body(..., embed=True),
):
    if rating < 1 or rating > 5:
        return JSONResponse(
            status_code=400,
            content={"error": "Rating must be between 1 and 5."},
        )

    # Call LLM
    try:
        summary, actions, user_response = generate_ai_analysis(rating, review)
    except Exception as e:
        # Fail soft, but still store basic feedback
        summary = "AI summary unavailable due to an error."
        actions = "Check the logs for LLM error and retry."
        user_response = (
            "Thanks for your feedback! There was a small glitch generating a detailed "
            "response, but we truly appreciate your time."
        )
        print("LLM error:", e)

    # Build item
    item = FeedbackItem(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat() + "Z",
        rating=rating,
        review=review,
        ai_summary=summary,
        ai_recommended_actions=actions,
        ai_user_response=user_response,
    )

    # Persist
    data = load_data()
    data.append(item.dict())
    save_data(data)

    # Return the user-facing response + maybe echo stored info
    return {
        "success": True,
        "user_response": user_response,
    }


@app.get("/api/feedback")
async def list_feedback():
    """
    Returns all feedback items for the admin dashboard.
    """
    data = load_data()
    return data
