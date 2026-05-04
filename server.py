"""
FastAPI Server for MCQ Generator

Provides REST API endpoints for generating MCQs and storing them in MongoDB Atlas.
Wraps the existing MCQGenerator without modifying its internal logic.
"""

import os
import uuid
import tempfile
import json
import re
import random
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from main import MCQGenerator
from database import get_async_database, close_async_database, ensure_mcq_indexes, COLLECTIONS
from models import (
    GenerateMCQResponse, MCQResponse, SessionResponse,
    MCQListResponse, SessionListResponse, HealthResponse,
    FlashcardFeedbackRequest, FlashcardFeedbackResponse,
    ChatSessionUpsert, ChatSessionResponse, ChatSessionListResponse, ChatSessionRename,
    AssignmentSave, AssignmentResponse, AssignmentListResponse,
    DashboardSummaryResponse, DashboardTotals, DashboardSlice, DashboardWeakTopic, DashboardRecentTopic,
    VideoSave, VideoResponse, VideoListResponse, VideoScene,
    CommunityPostCreate, CommunityPostResponse, CommunityPostListResponse,
    CommunityLikeRequest, CommunityCommentCreate, CommunityComment,
)
from pymongo.errors import DuplicateKeyError
from pymongo import ReturnDocument
from answer_grading import build_answer_grading
from nodes.assembler import export_mcqs_to_markdown
import learner_memory

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MCQ Generator API",
    description="Generate high-quality multiple-choice questions for calculus/integration topics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default configuration from environment
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3-8b-instruct")
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "1"))
FALLBACK_LLM_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER", "openai")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "meta-llama/llama-3-8b-instruct")
FALLBACK_BATCH_SIZE = int(os.getenv("FALLBACK_BATCH_SIZE", "3"))


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 100) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        value = float(raw) if raw else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, min(maximum, value))


def _resolve_user_id(form_user_id: Optional[str], header_user_id: Optional[str]) -> tuple[Optional[str], str]:
    header_val = (header_user_id or "").strip()
    if header_val:
        return header_val, "header"
    form_val = (form_user_id or "").strip()
    if form_val:
        return form_val, "form"
    return None, "none"


DEFAULT_RAG_TOP_K = _env_int("RAG_TOP_K", 8, minimum=1, maximum=50)
HUB_QUERY_RAG_TOP_K = _env_int("RAG_TOP_K_HUB_QUERY", 20, minimum=1, maximum=80)
DEFAULT_RAG_MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.1, minimum=0.0, maximum=1.0)

def _requested_mcq_count(query: Optional[str], default_count: int = 10) -> int:
    """
    Infer desired MCQ count from user query text.
    Examples: "give me 15 mcqs", "create 8 questions".
    """
    if not query:
        return default_count
    match = re.search(r"\b(\d{1,2})\s*(mcq|mcqs|questions?)\b", query.lower())
    if not match:
        return default_count
    count = int(match.group(1))
    return max(1, min(20, count))

def _parse_json_response(text: str):
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text, strict=False)


def _safe_topic_label(topic: Optional[str], sub_topic: Optional[str], chapter: Optional[str]) -> str:
    t = (topic or "").strip()
    st = (sub_topic or "").strip()
    ch = (chapter or "").strip()
    if t and st:
        return f"{t} / {st}"
    if t:
        return t
    if ch:
        return ch
    return "General"


def _build_dashboard_weak_topics(
    sessions: List[Dict[str, Any]],
    mcq_topic_counts: Dict[str, int],
) -> List[DashboardWeakTopic]:
    aggregate: Dict[str, Dict[str, float]] = {}

    for session in sessions:
        topic_label = _safe_topic_label(
            session.get("topic"),
            session.get("sub_topic"),
            session.get("chapter"),
        )
        summary = session.get("first_attempt_summary") or {}
        attempts = int(summary.get("total_questions") or 0)
        wrong = int(summary.get("total_wrong") or 0)
        hard_bias = float((session.get("difficulty_distribution") or {}).get("hard") or 0)
        hard_ratio = (hard_bias / attempts) if attempts else 0.0
        wrong_ratio = (wrong / attempts) if attempts else 0.0
        severity = (wrong_ratio * 0.75) + (hard_ratio * 0.25)

        if topic_label not in aggregate:
            aggregate[topic_label] = {
                "score": 0.0,
                "attempts": 0.0,
                "wrong_answers": 0.0,
            }

        aggregate[topic_label]["score"] += severity * max(attempts, 1)
        aggregate[topic_label]["attempts"] += attempts
        aggregate[topic_label]["wrong_answers"] += wrong

    weak_topics: List[DashboardWeakTopic] = []
    for topic_label, vals in aggregate.items():
        attempts = int(vals["attempts"])
        if attempts <= 0 and mcq_topic_counts.get(topic_label, 0) <= 0:
            continue
        weighted = vals["score"] / max(attempts, 1)
        # Small frequency prior so repeatedly seen topics surface.
        weighted += min(0.15, mcq_topic_counts.get(topic_label, 0) * 0.01)
        weak_topics.append(
            DashboardWeakTopic(
                topic=topic_label,
                score=round(weighted, 3),
                attempts=attempts,
                wrong_answers=int(vals["wrong_answers"]),
            )
        )

    weak_topics.sort(key=lambda x: (x.score, x.wrong_answers, x.attempts), reverse=True)
    return weak_topics[:8]


def _summarize_weak_concepts_for_memory(
    *,
    subject: str,
    chapter: str,
    generation_query: str,
    weak_concepts: List[str],
    correct: int,
    wrong: int,
    graded: int,
    last_score: float,
    model: str,
) -> dict:
    weak_lines = [w.strip() for w in weak_concepts if str(w).strip()]
    if not weak_lines:
        return {
            "summary": f"Learner currently shows mastery in {subject} / {chapter}. Keep medium-hard mixed practice.",
            "focus_areas": ["Sustain mastery with mixed concept reinforcement."],
            "common_mistakes": [],
            "next_mcq_guidance": ["Use slight variation and increase challenge gradually."],
        }

    llm = ChatOpenAI(
        model=model,
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    system_prompt = (
        "You are a learning-analytics summarizer for math MCQ practice.\n"
        "Return ONLY valid JSON with this shape:\n"
        "{\n"
        '  "summary": "single concise paragraph",\n'
        '  "focus_areas": ["...", "..."],\n'
        '  "common_mistakes": ["...", "..."],\n'
        '  "next_mcq_guidance": ["...", "..."]\n'
        "}\n"
        "Rules:\n"
        "- summary: 1 short paragraph, max 80 words\n"
        "- focus_areas: 3-6 concise items\n"
        "- common_mistakes: 2-5 concise items\n"
        "- next_mcq_guidance: 2-4 actionable generation directives\n"
        "- Keep statements specific and pedagogically useful."
    )
    weak_blob = "\n".join(f"- {w}" for w in weak_lines[:20])
    user_prompt = (
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"Generation query: {generation_query or 'n/a'}\n"
        f"Score stats: correct={correct}, wrong={wrong}, graded={graded}, score_percent={last_score}\n\n"
        "Weak concept observations:\n"
        f"{weak_blob}\n\n"
        "Generate the JSON now."
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    data = _parse_json_response(response.content)
    summary = " ".join(str(data.get("summary", "")).split()).strip()
    focus_areas = [str(x).strip() for x in (data.get("focus_areas") or []) if str(x).strip()]
    common_mistakes = [str(x).strip() for x in (data.get("common_mistakes") or []) if str(x).strip()]
    next_mcq_guidance = [str(x).strip() for x in (data.get("next_mcq_guidance") or []) if str(x).strip()]

    if not summary:
        summary = "Learner weak areas detected; prioritize conceptual remediation and step-by-step reasoning checks."
    if not focus_areas:
        focus_areas = ["Reinforce weak concepts from recent incorrect attempts."]
    if not next_mcq_guidance:
        next_mcq_guidance = ["Generate novel remedial items targeting the weakest ideas first."]

    return {
        "summary": summary[:1200],
        "focus_areas": focus_areas[:8],
        "common_mistakes": common_mistakes[:8],
        "next_mcq_guidance": next_mcq_guidance[:8],
    }


def _sync_prepend_learner_memory(
    path: str,
    subject: str,
    chapter: str,
    query: Optional[str],
    user_id: str,
    top_k: int = 8,
    min_score: float = DEFAULT_RAG_MIN_SCORE,
) -> dict:
    """Runs in thread: query Pinecone and prepend weak-focus + memory block to chapter markdown."""
    td, sd, tk, sk = learner_memory.topic_subtopic_from_strings(subject, chapter)
    qt = (query or "").strip() or f"{td} {sd}"
    bundle = learner_memory.query_memory_bundle(
        user_id,
        tk,
        sk,
        qt,
        top_k=top_k,
        min_score=min_score,
    )
    composed = learner_memory.compose_learner_prefix_for_generation(bundle)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if composed.strip() and (
        int(bundle.get("memory_hits") or 0) > 0 or (bundle.get("weak_focus_markdown") or "").strip()
    ):
        with open(path, "w", encoding="utf-8") as f:
            f.write(composed.strip() + "\n\n" + content)
    return {
        "memory_hits": bundle["memory_hits"],
        "summary_hits": bundle.get("summary_hits") or 0,
        "difficulty_hint": bundle["difficulty_hint"],
        "topic_key": tk,
        "sub_topic_key": sk,
        "learner_context": bundle.get("learner_context") or "",
        "weak_focus_markdown": bundle.get("weak_focus_markdown") or "",
        "lane_used": bundle.get("lane_used") or "",
        "retrieval_tier": bundle.get("retrieval_tier") or "none",
        "retrieval_trace": bundle.get("retrieval_trace") or [],
    }


async def _prepend_learner_memory_file(
    path: Optional[str],
    subject: str,
    chapter: str,
    query: Optional[str],
    user_id: Optional[str],
    top_k: int = 8,
    min_score: float = DEFAULT_RAG_MIN_SCORE,
) -> dict:
    if not path or not user_id or not learner_memory.is_configured():
        td, sd, tk, sk = learner_memory.topic_subtopic_from_strings(subject, chapter)
        return {
            "memory_hits": 0,
            "summary_hits": 0,
            "difficulty_hint": "medium",
            "topic_key": tk,
            "sub_topic_key": sk,
            "learner_context": "",
            "weak_focus_markdown": "",
            "lane_used": "",
            "retrieval_tier": "none",
            "retrieval_trace": [],
        }
    return await asyncio.to_thread(
        _sync_prepend_learner_memory,
        path,
        subject,
        chapter,
        query,
        user_id,
        top_k,
        min_score,
    )


def _normalize_openrouter_mcqs_payload(
    raw_mcqs: list,
    *,
    include_explanations: bool,
    max_items: int,
) -> List[dict]:
    """Turn model JSON `mcqs` array into the internal dict shape used by `/generate-mcqs`."""
    cap = max(1, min(20, int(max_items or 10)))
    normalized: List[dict] = []
    if not isinstance(raw_mcqs, list):
        return []
    for item in raw_mcqs:
        if len(normalized) >= cap:
            break
        if not isinstance(item, dict):
            continue
        options = item.get("options", {})
        if not isinstance(options, dict) or not all(k in options for k in ["a", "b", "c", "d"]):
            continue
        answer = str(item.get("correct_answer", "")).lower().strip()
        if answer not in ["a", "b", "c", "d"]:
            continue

        explanation = {"correct": item.get("correct_explanation", "Correct option by standard method.")}
        if include_explanations:
            for key in ["a", "b", "c", "d"]:
                explanation[key] = "Option analysis available in tutor mode."

        normalized.append(
            {
                "question_number": len(normalized) + 1,
                "concept_id": str(item.get("concept_id", f"query_concept_{len(normalized) + 1}")),
                "stem": str(item.get("stem", f"Generated question {len(normalized) + 1}")),
                "options": options,
                "correct_answer": answer,
                "explanation": explanation,
                "metadata": {
                    "difficulty": str(item.get("difficulty", "medium")),
                    "validation_score": 1.0,
                    "was_corrected": False,
                    "integral_type": "query_generated",
                },
            }
        )
    return normalized


def _openrouter_flashcards_single_call(
    *,
    subject: str,
    chapter: str,
    query: str,
    desired_count: int,
    model: str,
    include_explanations: bool,
    learner_context: str = "",
    difficulty_hint: str = "medium",
) -> List[dict]:
    """
    One OpenRouter-compatible ChatOpenAI completion → MCQ list (Hub / flash cards).
    No LangGraph.
    """
    n = max(1, min(20, int(desired_count)))
    llm = ChatOpenAI(
        model=model,
        temperature=0.35,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    system_prompt = (
        "You are an expert mathematics MCQ generator. "
        "Return ONLY valid JSON with this shape:\n"
        "{\n"
        '  "mcqs": [\n'
        "    {\n"
        '      "stem": "question text",\n'
        '      "options": {"a":"...", "b":"...", "c":"...", "d":"..."},\n'
        '      "correct_answer": "a|b|c|d",\n'
        '      "difficulty": "easy|medium|hard",\n'
        '      "concept_id": "short_id",\n'
        '      "correct_explanation": "why answer is correct"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Generate **exactly {n}** MCQs. Each must have exactly four options a–d and a valid correct_answer."
        " If learner summary context is provided, prioritize those focus areas and mistakes while keeping stems novel."
    )

    user_prompt = (
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"User query/topic: {query}\n"
        "Generate conceptual and calculation-based MCQs aligned with the chapter."
    )
    if (learner_context or "").strip():
        user_prompt += (
            f"\n\nPrior learner activity and summary context (do not repeat verbatim stems):\n{learner_context[:4000]}\n"
            "\nInstruction: Use retrieved focus_areas/common_mistakes/next_mcq_guidance to generate "
            "remedial questions that target misconceptions directly while varying wording and numbers."
            f"\nTarget difficulty bias for NEW items: **{difficulty_hint}**."
        )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    data = _parse_json_response(response.content)
    raw_mcqs = data.get("mcqs", [])
    if not isinstance(raw_mcqs, list):
        return []
    return _normalize_openrouter_mcqs_payload(
        raw_mcqs,
        include_explanations=include_explanations,
        max_items=n,
    )


def _generate_mcqs_from_query_direct(
    subject: str,
    chapter: str,
    query: str,
    model: str,
    include_explanations: bool,
    desired_count: int = 10,
    learner_context: str = "",
    difficulty_hint: str = "medium",
):
    """
    Direct query-to-MCQ fallback when the full pipeline returns zero questions.
    """
    return _openrouter_flashcards_single_call(
        subject=subject,
        chapter=chapter,
        query=query,
        desired_count=desired_count,
        model=model,
        include_explanations=include_explanations,
        learner_context=learner_context,
        difficulty_hint=difficulty_hint,
    )

def _generate_mock_mcqs(
    subject: str,
    chapter: str,
    requested_count: int = 10,
    include_explanations: bool = True
):
    """
    Guaranteed fallback set so frontend can always render flash cards.
    """
    topic = f"{subject} - {chapter}".strip(" -")
    templates = [
        ("What is the derivative of $x^2$?", {"a": "$2x$", "b": "$x$", "c": "$x^2$", "d": "$2$"}, "a"),
        ("Evaluate $\\int 2x\\,dx$.", {"a": "$x^2 + C$", "b": "$2x + C$", "c": "$x + C$", "d": "$2x^2 + C$"}, "a"),
        ("$\\sin^2 x + \\cos^2 x =$ ?", {"a": "$1$", "b": "$0$", "c": "$\\sin x$", "d": "$\\cos x$"}, "a"),
        ("Derivative of $\\ln x$ is:", {"a": "$1/x$", "b": "$x$", "c": "$\\ln x$", "d": "$e^x$"}, "a"),
        ("$\\int \\frac{1}{x}\\,dx =$", {"a": "$\\ln|x| + C$", "b": "$1/x + C$", "c": "$x + C$", "d": "$e^x + C$"}, "a"),
        ("$\\frac{d}{dx}(e^x) =$", {"a": "$e^x$", "b": "$xe^{x-1}$", "c": "$x^e$", "d": "$1$"}, "a"),
        ("For $f(x)=x^3$, $f'(x)=$", {"a": "$3x^2$", "b": "$x^2$", "c": "$3x$", "d": "$x^3$"}, "a"),
        ("$\\int \\cos x\\,dx =$", {"a": "$\\sin x + C$", "b": "$-\\sin x + C$", "c": "$\\cos x + C$", "d": "$-\\cos x + C$"}, "a"),
        ("$\\int \\sin x\\,dx =$", {"a": "$-\\cos x + C$", "b": "$\\cos x + C$", "c": "$\\sin x + C$", "d": "$-\\sin x + C$"}, "a"),
        ("Power rule: $\\int x^n\\,dx =$", {"a": "$\\frac{x^{n+1}}{n+1}+C$", "b": "$nx^{n-1}+C$", "c": "$x^{n-1}+C$", "d": "$\\frac{x^n}{n}+C$"}, "a"),
    ]
    generated = []
    for idx in range(1, requested_count + 1):
        stem, options, correct = random.choice(templates)
        explanation = {
            "correct": f"Mock fallback card for {topic}. Replace with model output when credits are available."
        }
        if include_explanations:
            for key in ["a", "b", "c", "d"]:
                explanation[key] = "Fallback explanation."
        generated.append({
            "question_number": idx,
            "concept_id": f"mock_{idx}",
            "stem": f"[Mock] {stem}",
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
            "metadata": {
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "validation_score": 1.0,
                "was_corrected": False,
                "integral_type": "mock_fallback",
            },
        })
    return generated


def _build_first_attempt_summary(mcqs: List[dict]) -> dict:
    total_questions = len(mcqs)
    total_correct = 0
    total_wrong = 0
    for mcq in mcqs:
        grading = mcq.get("answer_grading") or {}
        total_correct += len(grading.get("correct_keys") or [])
        total_wrong += len(grading.get("incorrect_keys") or [])

    return {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "captured_at": datetime.utcnow(),
        "source": "first_generation_attempt",
    }


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    await get_async_database()
    await ensure_mcq_indexes()
    print("[OK] FastAPI server started")
    print("[OK] MongoDB connection initialized")
    print("[OK] MongoDB indexes ensured")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    await close_async_database()
    print("[OK] MongoDB connection closed")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MCQ Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generate-mcqs",
            "list_sessions": "GET /sessions",
            "get_session": "GET /sessions/{session_id}",
            "list_mcqs": "GET /mcqs",
            "get_mcq": "GET /mcqs/{mcq_id}",
            "health": "GET /health",
            "flashcard_feedback": "POST /flashcard-feedback",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies API and database connectivity.
    """
    try:
        db = await get_async_database()
        # Test database connection
        await db.command("ping")
        
        return HealthResponse(
            status="healthy",
            database="connected",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")


@app.post("/generate-mcqs", response_model=GenerateMCQResponse, tags=["Generation"])
async def generate_mcqs(
    file: UploadFile = File(None, description="Optional input file (chapter.md or existing MCQs)"),
    subject: str = Form(..., description="Subject name (e.g., 'Calculus', 'Linear Algebra')"),
    chapter: str = Form(..., description="Chapter name (e.g., 'Chapter 3 - Definite Integrals')"),
    topic: Optional[str] = Form(None, description="Optional display topic; when set, overrides subject for storage/memory"),
    sub_topic: Optional[str] = Form(None, description="Optional display sub-topic; when set, overrides chapter for storage/memory"),
    query: Optional[str] = Form(None, description="Optional natural-language query/topic for on-the-fly generation"),
    input_type: str = Form("chapter", description="Type of input: 'chapter' or 'mcqs'"),
    include_explanations: bool = Form(True, description="Include explanations in MCQs"),
    user_id: Optional[str] = Form(None, description="Logged-in user id"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """
    Generate MCQs from uploaded file.
    
    This endpoint accepts a markdown file (chapter content or existing MCQs)
    and generates new MCQs using the LLM configuration from environment variables.
    
    The generation is synchronous - the endpoint returns after completion.
    All results are stored in MongoDB Atlas organized by subject and chapter.
    
    Configuration is read from .env file:
    - DEFAULT_LLM_PROVIDER (gemini/openai/anthropic)
    - DEFAULT_MODEL (model name)
    - DEFAULT_BATCH_SIZE (batch size)
    """
    
    # Validate source and mode
    if not file and not (query and query.strip()):
        raise HTTPException(status_code=400, detail="Provide either an uploaded markdown file or a query")
    if input_type not in ["chapter", "mcqs"]:
        raise HTTPException(status_code=400, detail="input_type must be 'chapter' or 'mcqs'")

    effective_subject = ((topic or "").strip() or (subject or "").strip())
    effective_chapter = ((sub_topic or "").strip() or (chapter or "").strip())
    if not effective_subject or not effective_chapter:
        raise HTTPException(status_code=400, detail="subject/topic and chapter/sub_topic are required")
    
    # Use configuration from environment variables
    llm_provider = DEFAULT_LLM_PROVIDER
    model = DEFAULT_MODEL
    batch_size = DEFAULT_BATCH_SIZE
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    resolved_user_id, identity_source = _resolve_user_id(user_id, x_user_id)
    desired_count = _requested_mcq_count(query, default_count=10)
    mem_stats: dict = {
        "memory_hits": 0,
        "summary_hits": 0,
        "difficulty_hint": "medium",
        "topic_key": "",
        "sub_topic_key": "",
        "learner_context": "",
        "weak_focus_markdown": "",
        "lane_used": "",
        "retrieval_tier": "none",
        "retrieval_trace": [],
    }
    # Hub flash cards: query-only, no uploaded file (same as frontend generateMCQsFromQuery).
    hub_query_mode = bool(file is None and query and query.strip())

    # Create temporary file to store uploaded content
    temp_file_path = None

    try:
        # Save uploaded file OR synthesize temporary markdown from query
        if file:
            if not file.filename.endswith('.md'):
                raise HTTPException(status_code=400, detail="File must be a markdown (.md) file")
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.md') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8') as temp_file:
                temp_file.write(f"# {effective_subject}\n\n## {effective_chapter}\n\n{query.strip()}\n")
                temp_file_path = temp_file.name
            input_type = "chapter"

        # For Hub flashcard generation, retrieve a wider memory set and let
        # learner_memory apply similarity threshold filtering.
        rag_top_k = HUB_QUERY_RAG_TOP_K if hub_query_mode else DEFAULT_RAG_TOP_K
        rag_min_score = DEFAULT_RAG_MIN_SCORE
        _mem = await _prepend_learner_memory_file(
            temp_file_path,
            effective_subject,
            effective_chapter,
            query,
            resolved_user_id,
            top_k=rag_top_k,
            min_score=rag_min_score,
        )
        mem_stats.update(_mem)

        print(f"\n{'='*60}")
        print(f"API REQUEST - Session ID: {session_id}")
        print(f"{'='*60}")
        print(f"Subject: {effective_subject}")
        print(f"Chapter: {effective_chapter}")
        print(f"Source: {file.filename if file else 'query'}")
        print(f"Input Type: {input_type}")
        print(f"Hub query-only (skip LangGraph): {hub_query_mode}")
        print(f"Identity source: {identity_source}")
        print(f"Resolved user_id: {resolved_user_id or '[none]'}")
        print(f"RAG top_k: {rag_top_k} | min_score: {rag_min_score}")
        print(f"RAG retrieval tier: {mem_stats.get('retrieval_tier') or 'none'}")
        print(f"RAG memory hits: {int(mem_stats.get('memory_hits') or 0)}")
        print(f"LLM: {llm_provider} - {model}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*60}\n")

        mcqs: List[dict] = []
        if hub_query_mode:
            # Single OpenRouter-style completion path (no MCQGenerator / LangGraph).
            llm_provider = DEFAULT_LLM_PROVIDER
            batch_size = 1
            learner_prefix = learner_memory.compose_learner_prefix_for_generation(mem_stats).strip()[:12000]
            difficulty_hint = mem_stats.get("difficulty_hint") or "medium"
            last_err: Optional[Exception] = None
            for attempt_model in (DEFAULT_MODEL, FALLBACK_MODEL):
                try:
                    mcqs = await asyncio.to_thread(
                        _openrouter_flashcards_single_call,
                        subject=effective_subject,
                        chapter=effective_chapter,
                        query=query.strip(),
                        desired_count=desired_count,
                        model=attempt_model,
                        include_explanations=include_explanations,
                        learner_context=learner_prefix,
                        difficulty_hint=difficulty_hint,
                    )
                    if mcqs:
                        model = attempt_model
                        break
                except Exception as ex:
                    last_err = ex
                    print(f"\n[Hub flashcards] OpenRouter call failed for model={attempt_model}: {ex}\n")
            if not mcqs:
                detail = (
                    "Hub flashcard generation produced no valid MCQs from the model response."
                    + (f" Last error: {last_err}" if last_err else "")
                )
                raise HTTPException(status_code=502, detail=detail)
        else:
            # Initialize MCQ Generator with specified configuration
            generator = MCQGenerator(
                llm_provider=llm_provider,
                model=model,
                batch_size=batch_size
            )

            # Generate MCQs (synchronous - waits for completion)
            # If provider fails (timeout, rate-limit, etc.), keep flowing to fallback cards.
            try:
                mcqs = generator.generate_from_file(
                    input_path=temp_file_path,
                    input_type=input_type,
                    output_path=None,  # We'll handle export separately
                    include_explanations=include_explanations
                )
            except Exception as primary_error:
                err_text = str(primary_error)
                print("\n" + "="*60)
                print("Primary provider request failed; attempting fallback provider")
                print(f"Primary error: {err_text}")
                print(f"Fallback LLM: {FALLBACK_LLM_PROVIDER} - {FALLBACK_MODEL}")
                print(f"Fallback batch size: {FALLBACK_BATCH_SIZE}")
                print("="*60 + "\n")

                llm_provider = FALLBACK_LLM_PROVIDER
                model = FALLBACK_MODEL
                batch_size = FALLBACK_BATCH_SIZE

                try:
                    fallback_generator = MCQGenerator(
                        llm_provider=llm_provider,
                        model=model,
                        batch_size=batch_size
                    )
                    mcqs = fallback_generator.generate_from_file(
                        input_path=temp_file_path,
                        input_type=input_type,
                        output_path=None,
                        include_explanations=include_explanations
                    )
                except Exception as fallback_error:
                    print(f"Fallback provider failed: {fallback_error}")
                    mcqs = []

            # Last-mile fallback: if pipeline produced no cards, generate directly from query.
            if len(mcqs) == 0 and query and query.strip():
                print("\n" + "="*60)
                print("Pipeline returned 0 MCQs; using direct query-based generation fallback")
                print("="*60 + "\n")
                try:
                    mcqs = _generate_mcqs_from_query_direct(
                        subject=effective_subject,
                        chapter=effective_chapter,
                        query=query.strip(),
                        model=FALLBACK_MODEL,
                        include_explanations=include_explanations,
                        desired_count=desired_count,
                        learner_context=learner_memory.compose_learner_prefix_for_generation(mem_stats).strip()[:12000],
                        difficulty_hint=mem_stats.get("difficulty_hint") or "medium",
                    )
                except Exception as direct_error:
                    print(f"Direct generation fallback failed: {direct_error}")
                    mcqs = []

            # Hard guarantee fallback for UI testing (not used for Hub query-only mode):
            # if still empty OR less than requested, pad using deterministic mock cards.
            if len(mcqs) == 0:
                print("\nUsing mock MCQ fallback (no model output available).")
                mcqs = _generate_mock_mcqs(
                    subject=effective_subject,
                    chapter=effective_chapter,
                    requested_count=desired_count,
                    include_explanations=include_explanations,
                )
            elif len(mcqs) < desired_count:
                print(f"\nTop-up fallback: generated {len(mcqs)}, requested {desired_count}.")
                extras = _generate_mock_mcqs(
                    subject=effective_subject,
                    chapter=effective_chapter,
                    requested_count=desired_count - len(mcqs),
                    include_explanations=include_explanations,
                )
                # Continue numbering and append mock extras
                start_idx = len(mcqs) + 1
                for i, extra in enumerate(extras, start=start_idx):
                    extra["question_number"] = i
                    extra["concept_id"] = f"mock_{i}"
                mcqs.extend(extras)
        
        # Deterministic grading payload for each option set
        for mcq in mcqs:
            mcq["answer_grading"] = build_answer_grading(mcq)

        # Calculate metrics
        difficulty_dist = {}
        for mcq in mcqs:
            diff = mcq['metadata']['difficulty']
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        
        metrics = {
            "total_concepts_extracted": len(mcqs),
            "total_mcqs_generated": len(mcqs),
            "validation_rate": 1.0,  # Since validator is pass-through
            "difficulty_distribution": difficulty_dist
        }
        
        # Generate markdown content
        markdown_output = []
        markdown_output.append(f"### Generated MCQs: Integration")
        markdown_output.append(f"#### PRACTICE EXERCISE")
        markdown_output.append(f"")
        
        from nodes.assembler import format_mcq_markdown
        for mcq in mcqs:
            mcq_text = format_mcq_markdown(mcq, include_explanations)
            markdown_output.append(mcq_text)
        
        markdown_content = "\n".join(markdown_output)
        
        # Persist session + MCQs in MongoDB.
        db = await get_async_database()
        now = datetime.utcnow()
        generation_query = query.strip() if query and query.strip() else None
        session_doc = {
            "session_id": session_id,
            "user_id": resolved_user_id,
            "subject": effective_subject,
            "chapter": effective_chapter,
            "generation_query": generation_query,
            "input_filename": file.filename if file else "query_input.md",
            "input_type": input_type,
            "llm_provider": llm_provider,
            "model": model,
            "batch_size": batch_size,
            "total_concepts_extracted": len(mcqs),
            "total_mcqs_generated": len(mcqs),
            "difficulty_distribution": difficulty_dist,
            "validation_rate": metrics.get("validation_rate", 1.0),
            "metrics": metrics,
            "first_attempt_summary": _build_first_attempt_summary(mcqs),
            "status": "completed",
            "error_message": None,
            "created_at": now,
            "completed_at": now,
        }
        await db[COLLECTIONS["mcq_sessions"]].update_one(
            {"session_id": session_id},
            {"$setOnInsert": session_doc},
            upsert=True,
        )

        mcq_docs = []
        mcq_responses = []
        _topic_val = (topic or "").strip() or None
        _sub_topic_val = (sub_topic or "").strip() or None
        for i, mcq in enumerate(mcqs, start=1):
            api_mcq_id = f"{session_id}_{i}"
            mcq_docs.append({
                "api_mcq_id": api_mcq_id,
                "session_id": session_id,
                "user_id": resolved_user_id,
                "subject": effective_subject,
                "chapter": effective_chapter,
                "topic": _topic_val,
                "sub_topic": _sub_topic_val,
                "generation_query": generation_query,
                "question_number": mcq.get("question_number", i),
                "concept_id": mcq.get("concept_id", f"concept_{i}"),
                "stem": mcq["stem"],
                "options": mcq["options"],
                "correct_answer": mcq["correct_answer"],
                "explanation": mcq["explanation"],
                "answer_grading": mcq.get("answer_grading"),
                "metadata": mcq["metadata"],
                "created_at": now,
            })
            mcq_responses.append(MCQResponse(
                id=api_mcq_id,
                session_id=session_id,
                user_id=resolved_user_id,
                subject=effective_subject,
                chapter=effective_chapter,
                topic=_topic_val,
                sub_topic=_sub_topic_val,
                question_number=mcq.get("question_number", i),
                concept_id=mcq.get("concept_id", f"concept_{i}"),
                stem=mcq["stem"],
                options=mcq["options"],
                correct_answer=mcq["correct_answer"],
                explanation=mcq["explanation"],
                answer_grading=mcq.get("answer_grading"),
                metadata=mcq["metadata"],
                created_at=now
            ))
        if mcq_docs:
            await db[COLLECTIONS["mcqs"]].insert_many(mcq_docs, ordered=True)

        memory_enabled = bool(resolved_user_id and learner_memory.is_configured())
        memory_hits = int(mem_stats.get("memory_hits") or 0)
        summary_hits = int(mem_stats.get("summary_hits") or 0)
        personalization_applied = bool(
            memory_enabled and (
                memory_hits > 0
                or (mem_stats.get("learner_context") or "").strip()
                or (mem_stats.get("weak_focus_markdown") or "").strip()
            )
        )
        if not resolved_user_id:
            personalization_reason = "missing_user_id"
        elif not learner_memory.is_configured():
            personalization_reason = "learner_memory_not_configured"
        elif personalization_applied:
            personalization_reason = "rag_context_applied"
        else:
            personalization_reason = "no_rag_hits"

        print(f"\n{'='*60}")
        print(f"[OK] Generation completed successfully!")
        print(f"[OK] Session ID: {session_id}")
        print(f"[OK] MCQs generated: {len(mcqs)}")
        print(f"[OK] Session and MCQs saved to MongoDB")
        print(f"{'='*60}\n")
        
        return GenerateMCQResponse(
            session_id=session_id,
            message="MCQs generated successfully",
            total_mcqs_generated=len(mcqs),
            difficulty_distribution=difficulty_dist,
            metrics=metrics,
            mcqs=mcq_responses,
            markdown_content=markdown_content,
            learner_memory_enabled=memory_enabled,
            memory_hits=memory_hits,
            summary_hits=summary_hits,
            difficulty_hint_applied=str(mem_stats.get("difficulty_hint") or "medium"),
            topic_key=(mem_stats.get("topic_key") or "") or None,
            sub_topic_key=(mem_stats.get("sub_topic_key") or "") or None,
            weak_focus_markdown=(mem_stats.get("weak_focus_markdown") or "").strip() or None,
            personalization_applied=personalization_applied,
            personalization_reason=personalization_reason,
            memory_retrieval_tier=(mem_stats.get("retrieval_tier") or "") or None,
            memory_lane_used=(mem_stats.get("lane_used") or "") or None,
            rag_top_k=rag_top_k,
            rag_min_score=rag_min_score,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Safety net for provider paths that still raise "no questions" upstream.
        # We degrade to deterministic mock cards instead of failing the API call.
        err_text = str(e)
        if hub_query_mode:
            print(f"\n{'='*60}")
            print(f"[ERR] Hub query-only generation failed!")
            print(f"[ERR] Session ID: {session_id}")
            print(f"[ERR] Error: {str(e)}")
            print(f"{'='*60}\n")
            raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")
        lowered = err_text.lower()
        if "produced no questions" in lowered or "mock fallback is disabled" in lowered:
            try:
                emergency_mcqs = _generate_mock_mcqs(
                    subject=effective_subject,
                    chapter=effective_chapter,
                    requested_count=desired_count,
                    include_explanations=include_explanations,
                )
                for mcq in emergency_mcqs:
                    mcq["answer_grading"] = build_answer_grading(mcq)

                emergency_dist = {}
                for mcq in emergency_mcqs:
                    diff = mcq["metadata"]["difficulty"]
                    emergency_dist[diff] = emergency_dist.get(diff, 0) + 1

                emergency_metrics = {
                    "total_concepts_extracted": len(emergency_mcqs),
                    "total_mcqs_generated": len(emergency_mcqs),
                    "validation_rate": 1.0,
                    "difficulty_distribution": emergency_dist,
                }

                markdown_output = [
                    "### Generated MCQs: Integration",
                    "#### PRACTICE EXERCISE",
                    "",
                ]
                from nodes.assembler import format_mcq_markdown
                for mcq in emergency_mcqs:
                    markdown_output.append(format_mcq_markdown(mcq, include_explanations))
                markdown_content = "\n".join(markdown_output)

                db = await get_async_database()
                now = datetime.utcnow()
                generation_query = query.strip() if query and query.strip() else None
                session_doc = {
                    "session_id": session_id,
                    "user_id": resolved_user_id,
                    "subject": effective_subject,
                    "chapter": effective_chapter,
                    "generation_query": generation_query,
                    "input_filename": file.filename if file else "query_input.md",
                    "input_type": input_type,
                    "llm_provider": llm_provider,
                    "model": model,
                    "batch_size": batch_size,
                    "total_concepts_extracted": len(emergency_mcqs),
                    "total_mcqs_generated": len(emergency_mcqs),
                    "difficulty_distribution": emergency_dist,
                    "validation_rate": emergency_metrics.get("validation_rate", 1.0),
                    "metrics": emergency_metrics,
                    "first_attempt_summary": _build_first_attempt_summary(emergency_mcqs),
                    "status": "completed_with_mock_fallback",
                    "error_message": err_text[:1200],
                    "created_at": now,
                    "completed_at": now,
                }
                await db[COLLECTIONS["mcq_sessions"]].update_one(
                    {"session_id": session_id},
                    {"$setOnInsert": session_doc},
                    upsert=True,
                )

                mcq_docs = []
                mcq_responses = []
                for i, mcq in enumerate(emergency_mcqs, start=1):
                    api_mcq_id = f"{session_id}_{i}"
                    mcq_docs.append({
                        "api_mcq_id": api_mcq_id,
                        "session_id": session_id,
                        "user_id": resolved_user_id,
                        "subject": effective_subject,
                        "chapter": effective_chapter,
                        "topic": _topic_val,
                        "sub_topic": _sub_topic_val,
                        "generation_query": generation_query,
                        "question_number": mcq.get("question_number", i),
                        "concept_id": mcq.get("concept_id", f"concept_{i}"),
                        "stem": mcq["stem"],
                        "options": mcq["options"],
                        "correct_answer": mcq["correct_answer"],
                        "explanation": mcq["explanation"],
                        "answer_grading": mcq.get("answer_grading"),
                        "metadata": mcq["metadata"],
                        "created_at": now,
                    })
                    mcq_responses.append(MCQResponse(
                        id=api_mcq_id,
                        session_id=session_id,
                        user_id=resolved_user_id,
                        subject=effective_subject,
                        chapter=effective_chapter,
                        topic=_topic_val,
                        sub_topic=_sub_topic_val,
                        question_number=mcq.get("question_number", i),
                        concept_id=mcq.get("concept_id", f"concept_{i}"),
                        stem=mcq["stem"],
                        options=mcq["options"],
                        correct_answer=mcq["correct_answer"],
                        explanation=mcq["explanation"],
                        answer_grading=mcq.get("answer_grading"),
                        metadata=mcq["metadata"],
                        created_at=now,
                    ))
                if mcq_docs:
                    await db[COLLECTIONS["mcqs"]].insert_many(mcq_docs, ordered=True)

                memory_enabled = bool(resolved_user_id and learner_memory.is_configured())
                memory_hits = int(mem_stats.get("memory_hits") or 0)
                summary_hits = int(mem_stats.get("summary_hits") or 0)
                personalization_applied = bool(
                    memory_enabled and (
                        memory_hits > 0
                        or (mem_stats.get("learner_context") or "").strip()
                        or (mem_stats.get("weak_focus_markdown") or "").strip()
                    )
                )
                if not resolved_user_id:
                    personalization_reason = "missing_user_id"
                elif not learner_memory.is_configured():
                    personalization_reason = "learner_memory_not_configured"
                elif personalization_applied:
                    personalization_reason = "rag_context_applied"
                else:
                    personalization_reason = "no_rag_hits"
                return GenerateMCQResponse(
                    session_id=session_id,
                    message="MCQs generated using mock fallback (provider returned no questions)",
                    total_mcqs_generated=len(emergency_mcqs),
                    difficulty_distribution=emergency_dist,
                    metrics=emergency_metrics,
                    mcqs=mcq_responses,
                    markdown_content=markdown_content,
                    learner_memory_enabled=memory_enabled,
                    memory_hits=memory_hits,
                    summary_hits=summary_hits,
                    difficulty_hint_applied=str(mem_stats.get("difficulty_hint") or "medium"),
                    topic_key=(mem_stats.get("topic_key") or "") or None,
                    sub_topic_key=(mem_stats.get("sub_topic_key") or "") or None,
                    weak_focus_markdown=(mem_stats.get("weak_focus_markdown") or "").strip() or None,
                    personalization_applied=personalization_applied,
                    personalization_reason=personalization_reason,
                    memory_retrieval_tier=(mem_stats.get("retrieval_tier") or "") or None,
                    memory_lane_used=(mem_stats.get("lane_used") or "") or None,
                    rag_top_k=rag_top_k,
                    rag_min_score=rag_min_score,
                )
            except Exception as fallback_error:
                print(f"[ERR] Emergency mock fallback failed: {fallback_error}")

        print(f"\n{'='*60}")
        print(f"[ERR] Generation failed!")
        print(f"[ERR] Session ID: {session_id}")
        print(f"[ERR] Error: {str(e)}")
        print(f"{'='*60}\n")
        
        raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/flashcard-feedback", response_model=FlashcardFeedbackResponse, tags=["Learner memory"])
async def flashcard_feedback(
    payload: FlashcardFeedbackRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    """
    Record flashcard practice outcomes and upsert a Pinecone vector with weak_concepts
    for later RAG during MCQ generation.
    """
    uid, _identity_source = _resolve_user_id(payload.user_id, x_user_id)
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required (form field or X-User-Id header)")

    db = await get_async_database()

    sess = await db[COLLECTIONS["mcq_sessions"]].find_one({"session_id": payload.session_id})
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    mcqs = (
        await db[COLLECTIONS["mcqs"]]
        .find({"session_id": payload.session_id})
        .sort("question_number", 1)
        .to_list(length=500)
    )
    if not mcqs:
        raise HTTPException(status_code=400, detail="No MCQs found for this session")

    su = (sess.get("user_id") or "").strip()
    if su and su != uid:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")
    if not su:
        mu = (mcqs[0].get("user_id") or "").strip()
        if mu and mu != uid:
            raise HTTPException(status_code=403, detail="MCQs in this session belong to a different user")

    by_id = {m["api_mcq_id"]: m for m in mcqs if m.get("api_mcq_id")}
    by_num = {int(m["question_number"]): m for m in mcqs if m.get("question_number") is not None}

    weak: List[str] = []
    wrong = 0
    correct = 0
    graded = 0

    for att in payload.attempts:
        m = None
        if att.mcq_id and att.mcq_id in by_id:
            m = by_id[att.mcq_id]
        elif att.question_number is not None and att.question_number in by_num:
            m = by_num[att.question_number]
        if not m:
            continue
        if not att.selected:
            continue
        graded += 1
        ca = str(m.get("correct_answer", "")).strip().lower()
        sel = att.selected.lower()
        if sel == ca:
            correct += 1
            continue
        wrong += 1
        stem_snip = re.sub(r"\s+", " ", (m.get("stem") or ""))[:120]
        concept_id_s = str(m.get("concept_id", ""))
        qn = m.get("question_number", "?")
        weak.append(f"Q{qn}: chose ({sel}) vs correct ({ca}); concept={concept_id_s}; stem: {stem_snip}")

    if graded == 0:
        raise HTTPException(
            status_code=400,
            detail="No graded attempts: include selected option for each attempted card",
        )

    weak = weak[:20]
    last_score = round(100.0 * correct / graded, 1)

    cid = (payload.client_event_id or "").strip()
    if cid:
        existing = await db[COLLECTIONS["flashcard_feedback_dedup"]].find_one(
            {"user_id": uid, "client_event_id": cid}
        )
        if existing:
            return FlashcardFeedbackResponse(
                message="Duplicate client_event_id; ignored.",
                weak_concepts_count=0,
                last_score=None,
                wrong_count=0,
                total_graded=0,
                pinecone_upserted=False,
                duplicate_event=True,
            )

    pinecone_ok = False
    memory_source = "disabled"
    summary_generated = False
    summary_length = 0
    weak_summary = ""
    focus_areas: List[str] = []
    common_mistakes: List[str] = []
    next_mcq_guidance: List[str] = []
    if learner_memory.is_configured():
        weak_excerpt = weak[0][:800] if weak else ""
        mastery_excerpt = (
            f"Session mastery summary: score={last_score}%, correct={correct}, "
            f"wrong={wrong}, graded={graded}"
        )
        try:
            summary_payload = await asyncio.to_thread(
                _summarize_weak_concepts_for_memory,
                subject=str(sess.get("subject") or ""),
                chapter=str(sess.get("chapter") or ""),
                generation_query=(sess.get("generation_query") or "").strip(),
                weak_concepts=weak,
                correct=correct,
                wrong=wrong,
                graded=graded,
                last_score=last_score,
                model=DEFAULT_MODEL,
            )
            weak_summary = str(summary_payload.get("summary") or "").strip()
            focus_areas = [str(x).strip() for x in (summary_payload.get("focus_areas") or []) if str(x).strip()]
            common_mistakes = [
                str(x).strip() for x in (summary_payload.get("common_mistakes") or []) if str(x).strip()
            ]
            next_mcq_guidance = [
                str(x).strip() for x in (summary_payload.get("next_mcq_guidance") or []) if str(x).strip()
            ]
            summary_generated = bool(weak_summary)
            summary_length = len(weak_summary)
        except Exception as summary_err:
            weak_summary = weak_excerpt or mastery_excerpt
            focus_areas = weak[:5]
            common_mistakes = weak[:5]
            next_mcq_guidance = ["Prioritize remediation for repeatedly missed sub-skills first."]
            summary_generated = False
            summary_length = len(weak_summary)
            print(f"[flashcard_feedback] summary generation failed: {summary_err}")
        excerpt_mem = weak_excerpt or mastery_excerpt
        prompt_mem = (sess.get("generation_query") or "").strip() or "flashcard_practice"
        source = "mcq_flashcard_practice_weak" if weak else "mcq_flashcard_practice_mastery"
        vid = await asyncio.to_thread(
            learner_memory.upsert_memory_record,
            user_id=uid,
            topic=str(sess.get("subject") or ""),
            sub_topic=str(sess.get("chapter") or ""),
            prompt=prompt_mem,
            source=source,
            session_id=payload.session_id,
            excerpt=excerpt_mem,
            weak_concepts=weak or [mastery_excerpt],
            weak_summary=weak_summary,
            focus_areas=focus_areas,
            common_mistakes=common_mistakes,
            next_mcq_guidance=next_mcq_guidance,
            last_score=last_score,
            wrong_count=wrong,
            total_attempted=graded,
        )
        pinecone_ok = bool(vid)
        memory_source = source
    elif weak:
        memory_source = "learner_memory_not_configured"
    else:
        memory_source = "mastery_not_persisted_memory_disabled"

    if cid:
        try:
            await db[COLLECTIONS["flashcard_feedback_dedup"]].insert_one(
                {"user_id": uid, "client_event_id": cid, "created_at": datetime.utcnow()}
            )
        except DuplicateKeyError:
            pass

    return FlashcardFeedbackResponse(
        message="Flashcard feedback recorded",
        weak_concepts_count=len(weak),
        last_score=last_score,
        wrong_count=wrong,
        total_graded=graded,
        pinecone_upserted=pinecone_ok,
        duplicate_event=False,
        memory_source=memory_source,
        summary_generated=summary_generated,
        summary_length=summary_length,
    )


@app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(
    subject: Optional[str] = Query(None, description="Filter by subject"),
    chapter: Optional[str] = Query(None, description="Filter by chapter"),
    session_id: Optional[str] = Query(None, description="Filter by exact session_id (deep link)"),
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum sessions to return")
):
    """
    List all MCQ generation sessions.
    
    Optionally filter by subject, chapter, and/or session_id. Returns paginated list of sessions with metadata.
    """
    db = await get_async_database()
    
    # Build query filter
    query_filter = {}
    if subject:
        query_filter["subject"] = subject
    if chapter:
        query_filter["chapter"] = chapter
    if session_id and session_id.strip():
        query_filter["session_id"] = session_id.strip()
    
    # Get total count
    total = await db[COLLECTIONS["mcq_sessions"]].count_documents(query_filter)
    
    # Fetch sessions
    sessions = await db[COLLECTIONS["mcq_sessions"]].find(query_filter)\
        .sort("created_at", -1)\
        .skip(skip)\
        .limit(limit)\
        .to_list(length=limit)
    
    session_responses = []
    for session in sessions:
        session_responses.append(SessionResponse(
            id=str(session["_id"]),
            session_id=session["session_id"],
            user_id=session.get("user_id"),
            subject=session["subject"],
            chapter=session["chapter"],
            generation_query=session.get("generation_query"),
            input_filename=session["input_filename"],
            input_type=session["input_type"],
            llm_provider=session["llm_provider"],
            model=session["model"],
            total_concepts_extracted=session["total_concepts_extracted"],
            total_mcqs_generated=session["total_mcqs_generated"],
            difficulty_distribution=session["difficulty_distribution"],
            first_attempt_summary=session.get("first_attempt_summary"),
            status=session["status"],
            created_at=session["created_at"],
            completed_at=session.get("completed_at")
        ))
    
    return SessionListResponse(
        total=total,
        sessions=session_responses
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def get_session(session_id: str):
    """
    Get details of a specific MCQ generation session.
    """
    db = await get_async_database()
    
    session = await db[COLLECTIONS["mcq_sessions"]].find_one({"session_id": session_id})
        
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=str(session["_id"]),
        session_id=session["session_id"],
        user_id=session.get("user_id"),
        subject=session["subject"],
        chapter=session["chapter"],
        generation_query=session.get("generation_query"),
        input_filename=session["input_filename"],
        input_type=session["input_type"],
        llm_provider=session["llm_provider"],
        model=session["model"],
        total_concepts_extracted=session["total_concepts_extracted"],
        total_mcqs_generated=session["total_mcqs_generated"],
        difficulty_distribution=session["difficulty_distribution"],
        first_attempt_summary=session.get("first_attempt_summary"),
        status=session["status"],
        created_at=session["created_at"],
        completed_at=session.get("completed_at")
    )


@app.get("/mcqs", response_model=MCQListResponse, tags=["MCQs"])
async def list_mcqs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    chapter: Optional[str] = Query(None, description="Filter by chapter"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    sub_topic: Optional[str] = Query(None, description="Filter by sub-topic"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (easy, medium, hard)"),
    skip: int = Query(0, ge=0, description="Number of MCQs to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum MCQs to return")
):
    """
    List generated MCQs with optional filters including user_id, subject, chapter, topic, sub_topic.
    """
    db = await get_async_database()

    query_filter = {}
    if user_id:
        query_filter["user_id"] = user_id
    if subject:
        query_filter["subject"] = subject
    if chapter:
        query_filter["chapter"] = chapter
    if topic:
        query_filter["topic"] = topic
    if sub_topic:
        query_filter["sub_topic"] = sub_topic
    if session_id:
        query_filter["session_id"] = session_id
    if difficulty:
        query_filter["metadata.difficulty"] = difficulty

    total = await db[COLLECTIONS["mcqs"]].count_documents(query_filter)

    mcqs = await db[COLLECTIONS["mcqs"]].find(query_filter)\
        .sort([("created_at", -1), ("question_number", 1)])\
        .skip(skip)\
        .limit(limit)\
        .to_list(length=limit)

    mcq_responses = []
    for mcq in mcqs:
        mcq_responses.append(MCQResponse(
            id=str(mcq.get("api_mcq_id") or str(mcq["_id"])),
            session_id=mcq["session_id"],
            user_id=mcq.get("user_id"),
            subject=mcq["subject"],
            chapter=mcq["chapter"],
            topic=mcq.get("topic"),
            sub_topic=mcq.get("sub_topic"),
            question_number=mcq["question_number"],
            concept_id=mcq["concept_id"],
            stem=mcq["stem"],
            options=mcq["options"],
            correct_answer=mcq["correct_answer"],
            explanation=mcq["explanation"],
            answer_grading=mcq.get("answer_grading"),
            metadata=mcq["metadata"],
            created_at=mcq["created_at"]
        ))

    return MCQListResponse(
        total=total,
        mcqs=mcq_responses
    )


@app.get("/subjects", response_model=dict, tags=["Subjects"])
async def list_subjects():
    """
    Get list of all unique subjects in the database.
    
    Returns a list of subjects with their MCQ counts.
    """
    db = await get_async_database()
    
    # Get distinct subjects from sessions collection
    subjects = await db[COLLECTIONS["mcq_sessions"]].distinct("subject")
    
    # Get counts for each subject
    subject_stats = []
    for subject in subjects:
        session_count = await db[COLLECTIONS["mcq_sessions"]].count_documents({"subject": subject})
        mcq_count = await db[COLLECTIONS["mcqs"]].count_documents({"subject": subject})
        subject_stats.append({
            "subject": subject,
            "total_sessions": session_count,
            "total_mcqs": mcq_count
        })
    
    # Sort by subject name
    subject_stats.sort(key=lambda x: x["subject"])
    
    return {
        "total_subjects": len(subjects),
        "subjects": subject_stats
    }


@app.get("/mcqs/{mcq_id}", response_model=MCQResponse, tags=["MCQs"])
async def get_mcq(mcq_id: str):
    """
    Get details of a specific MCQ.
    """
    from bson import ObjectId
    
    db = await get_async_database()
    
    try:
        mcq = await db[COLLECTIONS["mcqs"]].find_one({"_id": ObjectId(mcq_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid MCQ ID format")
    
    if not mcq:
        raise HTTPException(status_code=404, detail="MCQ not found")
    
    return MCQResponse(
        id=str(mcq.get("api_mcq_id") or str(mcq["_id"])),
        session_id=mcq["session_id"],
        user_id=mcq.get("user_id"),
        subject=mcq["subject"],
        chapter=mcq["chapter"],
        topic=mcq.get("topic"),
        sub_topic=mcq.get("sub_topic"),
        question_number=mcq["question_number"],
        concept_id=mcq["concept_id"],
        stem=mcq["stem"],
        options=mcq["options"],
        correct_answer=mcq["correct_answer"],
        explanation=mcq["explanation"],
        answer_grading=mcq.get("answer_grading"),
        metadata=mcq["metadata"],
        created_at=mcq["created_at"]
    )


# --- Vercel serverless handler ---
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # mangum not needed when running locally with uvicorn
# ============================================================================
# Chat Session Endpoints
# ============================================================================

@app.post("/chat-sessions", response_model=ChatSessionResponse, tags=["Chat Sessions"])
async def upsert_chat_session(body: ChatSessionUpsert):
    """Create or update a chat session for a user (upsert by session_id)."""
    db = await get_async_database()
    now = datetime.utcnow()
    col = db[COLLECTIONS["chat_sessions"]]

    doc = {
        "session_id": body.session_id,
        "user_id": body.user_id,
        "title": body.title,
        "messages": body.messages,
        "updated_at": now,
    }
    result = await col.find_one_and_update(
        {"session_id": body.session_id, "user_id": body.user_id},
        {"$set": {k: v for k, v in doc.items() if k != "session_id"},
         "$setOnInsert": {"created_at": now}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if result is None:
        result = await col.find_one({"session_id": body.session_id, "user_id": body.user_id})

    return ChatSessionResponse(
        session_id=result["session_id"],
        user_id=result["user_id"],
        title=result["title"],
        messages=result.get("messages", []),
        created_at=result.get("created_at", now),
        updated_at=result.get("updated_at", now),
    )


@app.get("/chat-sessions", response_model=ChatSessionListResponse, tags=["Chat Sessions"])
async def list_chat_sessions(
    user_id: str = Query(..., description="User ID to fetch sessions for"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all chat sessions for a user, newest first."""
    db = await get_async_database()
    col = db[COLLECTIONS["chat_sessions"]]
    total = await col.count_documents({"user_id": user_id})
    docs = await col.find({"user_id": user_id})\
        .sort("updated_at", -1)\
        .skip(skip).limit(limit).to_list(length=limit)

    sessions = [
        ChatSessionResponse(
            session_id=d["session_id"],
            user_id=d["user_id"],
            title=d.get("title", "Chat"),
            messages=d.get("messages", []),
            created_at=d.get("created_at", d.get("updated_at", datetime.utcnow())),
            updated_at=d.get("updated_at", datetime.utcnow()),
        )
        for d in docs
    ]
    return ChatSessionListResponse(total=total, sessions=sessions)


@app.get("/chat-sessions/{session_id}", response_model=ChatSessionResponse, tags=["Chat Sessions"])
async def get_chat_session(session_id: str, user_id: str = Query(...)):
    """Get a single chat session by session_id (must belong to requesting user)."""
    db = await get_async_database()
    doc = await db[COLLECTIONS["chat_sessions"]].find_one(
        {"session_id": session_id, "user_id": user_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Chat session not found")
    now = datetime.utcnow()
    return ChatSessionResponse(
        session_id=doc["session_id"],
        user_id=doc["user_id"],
        title=doc.get("title", "Chat"),
        messages=doc.get("messages", []),
        created_at=doc.get("created_at", now),
        updated_at=doc.get("updated_at", now),
    )


@app.patch("/chat-sessions/{session_id}", response_model=ChatSessionResponse, tags=["Chat Sessions"])
async def rename_chat_session(session_id: str, body: ChatSessionRename):
    """Rename an existing chat session title for a user."""
    db = await get_async_database()
    now = datetime.utcnow()
    col = db[COLLECTIONS["chat_sessions"]]
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    updated = await col.find_one_and_update(
        {"session_id": session_id, "user_id": body.user_id},
        {"$set": {"title": title[:120], "updated_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return ChatSessionResponse(
        session_id=updated["session_id"],
        user_id=updated["user_id"],
        title=updated.get("title", "Chat"),
        messages=updated.get("messages", []),
        created_at=updated.get("created_at", now),
        updated_at=updated.get("updated_at", now),
    )


@app.delete("/chat-sessions/{session_id}", tags=["Chat Sessions"])
async def delete_chat_session(session_id: str, user_id: str = Query(...)):
    """Delete a chat session for a user."""
    db = await get_async_database()
    col = db[COLLECTIONS["chat_sessions"]]
    result = await col.delete_one({"session_id": session_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"ok": True, "deleted": session_id}


# ============================================================================
# Assignment Endpoints
# ============================================================================

@app.post("/assignments", response_model=AssignmentResponse, tags=["Assignments"])
async def save_assignment(body: AssignmentSave):
    """Save assignment text content for a user."""
    db = await get_async_database()
    now = datetime.utcnow()
    assignment_id = str(uuid.uuid4())
    doc = {
        "assignment_id": assignment_id,
        "user_id": body.user_id,
        "query": body.query,
        "title": body.title,
        "content": body.content,
        "subject": body.subject,
        "created_at": now,
    }
    await db[COLLECTIONS["user_assignments"]].insert_one(doc)
    return AssignmentResponse(
        assignment_id=assignment_id,
        user_id=body.user_id,
        query=body.query,
        title=body.title,
        content=body.content,
        subject=body.subject,
        created_at=now,
    )


@app.get("/assignments", response_model=AssignmentListResponse, tags=["Assignments"])
async def list_assignments(
    user_id: str = Query(..., description="User ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all saved assignments for a user, newest first."""
    db = await get_async_database()
    col = db[COLLECTIONS["user_assignments"]]
    total = await col.count_documents({"user_id": user_id})
    docs = await col.find({"user_id": user_id})\
        .sort("created_at", -1)\
        .skip(skip).limit(limit).to_list(length=limit)

    assignments = [
        AssignmentResponse(
            assignment_id=d["assignment_id"],
            user_id=d["user_id"],
            query=d["query"],
            title=d.get("title", d["query"]),
            content=d["content"],
            subject=d.get("subject"),
            created_at=d["created_at"],
        )
        for d in docs
    ]
    return AssignmentListResponse(total=total, assignments=assignments)


@app.get("/assignments/{assignment_id}", response_model=AssignmentResponse, tags=["Assignments"])
async def get_assignment(assignment_id: str, user_id: str = Query(...)):
    """Get a single saved assignment by ID (must belong to requesting user)."""
    db = await get_async_database()
    doc = await db[COLLECTIONS["user_assignments"]].find_one(
        {"assignment_id": assignment_id, "user_id": user_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return AssignmentResponse(
        assignment_id=doc["assignment_id"],
        user_id=doc["user_id"],
        query=doc["query"],
        title=doc.get("title", doc["query"]),
        content=doc["content"],
        subject=doc.get("subject"),
        created_at=doc["created_at"],
    )


@app.get("/dashboard/summary", response_model=DashboardSummaryResponse, tags=["Dashboard"])
async def dashboard_summary(user_id: str = Query(..., description="User ID")):
    """
    Aggregated personalized dashboard payload for learner UI.
    Includes totals, weak topics, recent topics, and chart-friendly distributions.
    """
    db = await get_async_database()
    sessions_col = db[COLLECTIONS["mcq_sessions"]]
    mcqs_col = db[COLLECTIONS["mcqs"]]
    assignments_col = db[COLLECTIONS["user_assignments"]]
    chats_col = db[COLLECTIONS["chat_sessions"]]

    total_mcqs = await mcqs_col.count_documents({"user_id": user_id})
    total_assignments = await assignments_col.count_documents({"user_id": user_id})
    total_chat_sessions = await chats_col.count_documents({"user_id": user_id})

    diff_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$metadata.difficulty", "value": {"$sum": 1}}},
    ]
    diff_rows = await mcqs_col.aggregate(diff_pipeline).to_list(length=20)
    diff_map = {str(r.get("_id") or "unknown"): int(r.get("value") or 0) for r in diff_rows}
    difficulty_distribution = [
        DashboardSlice(label="easy", value=diff_map.get("easy", 0)),
        DashboardSlice(label="medium", value=diff_map.get("medium", 0)),
        DashboardSlice(label="hard", value=diff_map.get("hard", 0)),
    ]

    subject_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$subject", "value": {"$sum": 1}}},
        {"$sort": {"value": -1}},
        {"$limit": 8},
    ]
    subject_rows = await mcqs_col.aggregate(subject_pipeline).to_list(length=8)
    subject_distribution = [
        DashboardSlice(label=str(r.get("_id") or "Unknown"), value=int(r.get("value") or 0))
        for r in subject_rows
    ]

    recent_mcqs = await mcqs_col.find({"user_id": user_id}) \
        .sort("created_at", -1).limit(120).to_list(length=120)

    recent_topics_map: Dict[str, Dict[str, Any]] = {}
    mcq_topic_counts: Dict[str, int] = {}
    for d in recent_mcqs:
        topic_label = _safe_topic_label(d.get("topic"), d.get("sub_topic"), d.get("chapter"))
        mcq_topic_counts[topic_label] = mcq_topic_counts.get(topic_label, 0) + 1
        if topic_label not in recent_topics_map:
            recent_topics_map[topic_label] = {
                "topic": d.get("topic") or d.get("chapter") or "General",
                "sub_topic": d.get("sub_topic"),
                "last_seen": d.get("created_at") or datetime.utcnow(),
                "mcq_count": 0,
            }
        recent_topics_map[topic_label]["mcq_count"] += 1
        if (d.get("created_at") or datetime.utcnow()) > recent_topics_map[topic_label]["last_seen"]:
            recent_topics_map[topic_label]["last_seen"] = d.get("created_at") or datetime.utcnow()

    recent_topics = [
        DashboardRecentTopic(
            topic=v["topic"],
            sub_topic=v.get("sub_topic"),
            last_seen=v["last_seen"],
            mcq_count=int(v["mcq_count"]),
        )
        for v in sorted(recent_topics_map.values(), key=lambda x: x["last_seen"], reverse=True)[:8]
    ]

    sessions = await sessions_col.find({"user_id": user_id}) \
        .sort("created_at", -1).limit(120).to_list(length=120)
    weak_topics = _build_dashboard_weak_topics(sessions, mcq_topic_counts)
    if not weak_topics:
        # Fallback: if no session-level attempt summaries, derive from topic frequency only.
        for topic_label, count in sorted(mcq_topic_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
            weak_topics.append(
                DashboardWeakTopic(topic=topic_label, score=round(min(1.0, count / 10), 3), attempts=count, wrong_answers=0)
            )

    return DashboardSummaryResponse(
        user_id=user_id,
        generated_at=datetime.utcnow(),
        totals=DashboardTotals(
            total_mcqs=total_mcqs,
            total_assignments=total_assignments,
            total_chat_sessions=total_chat_sessions,
        ),
        difficulty_distribution=difficulty_distribution,
        subject_distribution=subject_distribution,
        weak_topics=weak_topics,
        recent_topics=recent_topics,
    )


# ============================================================================
# Video Library Endpoints
# ============================================================================

@app.post("/videos", response_model=VideoResponse, tags=["Videos"])
async def save_video(body: VideoSave):
    """Save a Manim video session (scenes + query) for a user."""
    db = await get_async_database()
    now = datetime.utcnow()
    video_id = str(uuid.uuid4())
    doc = {
        "video_id": video_id,
        "user_id": body.user_id,
        "title": body.title,
        "original_query": body.original_query,
        "scenes": [s.dict() for s in body.scenes],
        "created_at": now,
    }
    await db[COLLECTIONS["user_videos"]].insert_one(doc)
    return VideoResponse(
        video_id=video_id,
        user_id=body.user_id,
        title=body.title,
        original_query=body.original_query,
        scenes=body.scenes,
        created_at=now,
    )


@app.get("/videos", response_model=VideoListResponse, tags=["Videos"])
async def list_videos(
    user_id: str = Query(..., description="User ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all saved video sessions for a user, newest first."""
    db = await get_async_database()
    col = db[COLLECTIONS["user_videos"]]
    total = await col.count_documents({"user_id": user_id})
    docs = await col.find({"user_id": user_id}) \
        .sort("created_at", -1).skip(skip).limit(limit).to_list(length=limit)
    videos = [
        VideoResponse(
            video_id=d["video_id"],
            user_id=d["user_id"],
            title=d.get("title", d["original_query"]),
            original_query=d["original_query"],
            scenes=[VideoScene(**s) for s in d.get("scenes", [])],
            created_at=d["created_at"],
        )
        for d in docs
    ]
    return VideoListResponse(total=total, videos=videos)


@app.get("/videos/{video_id}", response_model=VideoResponse, tags=["Videos"])
async def get_video(video_id: str, user_id: str = Query(...)):
    """Get a single saved video session by ID."""
    db = await get_async_database()
    doc = await db[COLLECTIONS["user_videos"]].find_one(
        {"video_id": video_id, "user_id": user_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(
        video_id=doc["video_id"],
        user_id=doc["user_id"],
        title=doc.get("title", doc["original_query"]),
        original_query=doc["original_query"],
        scenes=[VideoScene(**s) for s in doc.get("scenes", [])],
        created_at=doc["created_at"],
    )


@app.delete("/videos/{video_id}", tags=["Videos"])
async def delete_video(video_id: str, user_id: str = Query(...)):
    """Delete a saved video session."""
    db = await get_async_database()
    result = await db[COLLECTIONS["user_videos"]].delete_one(
        {"video_id": video_id, "user_id": user_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"ok": True, "deleted": video_id}


# ============================================================================
# Community Endpoints
# ============================================================================

@app.post("/community/posts", response_model=CommunityPostResponse, tags=["Community"])
async def create_community_post(body: CommunityPostCreate):
    """Share a post to the community."""
    db = await get_async_database()
    now = datetime.utcnow()
    post_id = str(uuid.uuid4())
    doc = {
        "post_id": post_id,
        "user_id": body.user_id,
        "author_name": body.author_name,
        "post_type": body.post_type,
        "title": body.title,
        "content": body.content,
        "topic": body.topic,
        "sub_topic": body.sub_topic,
        "likes": [],
        "dislikes": [],
        "comments": [],
        "created_at": now,
    }
    await db[COLLECTIONS["community_posts"]].insert_one(doc)
    return CommunityPostResponse(**{k: v for k, v in doc.items() if k != "_id"})


@app.get("/community/posts", response_model=CommunityPostListResponse, tags=["Community"])
async def list_community_posts(
    post_type: Optional[str] = Query(None, description="Filter by type: mcq, assignment, video"),
    topic: Optional[str] = Query(None),
    sub_topic: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Free-text search"),
    sort_by: str = Query("date", description="Sort field: date or likes"),
    order: str = Query("desc", description="asc or desc"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """List community posts with optional filtering and sorting."""
    db = await get_async_database()
    col = db[COLLECTIONS["community_posts"]]

    query_filter: Dict[str, Any] = {}
    if post_type:
        query_filter["post_type"] = post_type
    if topic:
        query_filter["topic"] = {"$regex": topic, "$options": "i"}
    if sub_topic:
        query_filter["sub_topic"] = {"$regex": sub_topic, "$options": "i"}
    if q:
        query_filter["$text"] = {"$search": q}

    sort_dir = -1 if order == "desc" else 1
    if sort_by == "likes":
        sort_field = [("likes_count", sort_dir), ("created_at", -1)]
        pipeline = [
            {"$match": query_filter},
            {"$addFields": {"likes_count": {"$size": "$likes"}}},
            {"$sort": {"likes_count": sort_dir, "created_at": -1}},
            {"$skip": skip},
            {"$limit": limit},
        ]
        count_pipeline = [{"$match": query_filter}, {"$count": "total"}]
        count_result = await col.aggregate(count_pipeline).to_list(length=1)
        total = count_result[0]["total"] if count_result else 0
        docs = await col.aggregate(pipeline).to_list(length=limit)
    else:
        total = await col.count_documents(query_filter)
        docs = await col.find(query_filter) \
            .sort("created_at", sort_dir).skip(skip).limit(limit).to_list(length=limit)

    posts = []
    for d in docs:
        posts.append(CommunityPostResponse(
            post_id=d["post_id"],
            user_id=d["user_id"],
            author_name=d.get("author_name", ""),
            post_type=d["post_type"],
            title=d.get("title", ""),
            content=d.get("content", ""),
            topic=d.get("topic", ""),
            sub_topic=d.get("sub_topic", ""),
            likes=d.get("likes", []),
            dislikes=d.get("dislikes", []),
            comments=[CommunityComment(**c) for c in d.get("comments", [])],
            created_at=d["created_at"],
        ))
    return CommunityPostListResponse(total=total, posts=posts)


@app.get("/community/posts/{post_id}", response_model=CommunityPostResponse, tags=["Community"])
async def get_community_post(post_id: str):
    """Get a single community post."""
    db = await get_async_database()
    doc = await db[COLLECTIONS["community_posts"]].find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Post not found")
    return CommunityPostResponse(
        post_id=doc["post_id"],
        user_id=doc["user_id"],
        author_name=doc.get("author_name", ""),
        post_type=doc["post_type"],
        title=doc.get("title", ""),
        content=doc.get("content", ""),
        topic=doc.get("topic", ""),
        sub_topic=doc.get("sub_topic", ""),
        likes=doc.get("likes", []),
        dislikes=doc.get("dislikes", []),
        comments=[CommunityComment(**c) for c in doc.get("comments", [])],
        created_at=doc["created_at"],
    )


@app.post("/community/posts/{post_id}/like", response_model=CommunityPostResponse, tags=["Community"])
async def toggle_like_post(post_id: str, body: CommunityLikeRequest):
    """Toggle like on a community post. Removes dislike if present."""
    db = await get_async_database()
    col = db[COLLECTIONS["community_posts"]]
    doc = await col.find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Post not found")

    uid = body.user_id
    likes = doc.get("likes", [])
    dislikes = [d for d in doc.get("dislikes", []) if d != uid]
    if uid in likes:
        likes = [l for l in likes if l != uid]
    else:
        likes = [l for l in likes if l != uid] + [uid]

    updated = await col.find_one_and_update(
        {"post_id": post_id},
        {"$set": {"likes": likes, "dislikes": dislikes}},
        return_document=ReturnDocument.AFTER,
    )
    return CommunityPostResponse(
        post_id=updated["post_id"], user_id=updated["user_id"],
        author_name=updated.get("author_name", ""), post_type=updated["post_type"],
        title=updated.get("title", ""), content=updated.get("content", ""),
        topic=updated.get("topic", ""), sub_topic=updated.get("sub_topic", ""),
        likes=updated.get("likes", []), dislikes=updated.get("dislikes", []),
        comments=[CommunityComment(**c) for c in updated.get("comments", [])],
        created_at=updated["created_at"],
    )


@app.post("/community/posts/{post_id}/dislike", response_model=CommunityPostResponse, tags=["Community"])
async def toggle_dislike_post(post_id: str, body: CommunityLikeRequest):
    """Toggle dislike on a community post. Removes like if present."""
    db = await get_async_database()
    col = db[COLLECTIONS["community_posts"]]
    doc = await col.find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Post not found")

    uid = body.user_id
    dislikes = doc.get("dislikes", [])
    likes = [l for l in doc.get("likes", []) if l != uid]
    if uid in dislikes:
        dislikes = [d for d in dislikes if d != uid]
    else:
        dislikes = [d for d in dislikes if d != uid] + [uid]

    updated = await col.find_one_and_update(
        {"post_id": post_id},
        {"$set": {"likes": likes, "dislikes": dislikes}},
        return_document=ReturnDocument.AFTER,
    )
    return CommunityPostResponse(
        post_id=updated["post_id"], user_id=updated["user_id"],
        author_name=updated.get("author_name", ""), post_type=updated["post_type"],
        title=updated.get("title", ""), content=updated.get("content", ""),
        topic=updated.get("topic", ""), sub_topic=updated.get("sub_topic", ""),
        likes=updated.get("likes", []), dislikes=updated.get("dislikes", []),
        comments=[CommunityComment(**c) for c in updated.get("comments", [])],
        created_at=updated["created_at"],
    )


@app.post("/community/posts/{post_id}/comments", response_model=CommunityPostResponse, tags=["Community"])
async def add_comment(post_id: str, body: CommunityCommentCreate):
    """Add a comment to a community post."""
    db = await get_async_database()
    col = db[COLLECTIONS["community_posts"]]
    doc = await col.find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Post not found")

    comment = {
        "comment_id": str(uuid.uuid4()),
        "user_id": body.user_id,
        "author_name": body.author_name,
        "body": body.body,
        "created_at": datetime.utcnow(),
    }
    updated = await col.find_one_and_update(
        {"post_id": post_id},
        {"$push": {"comments": comment}},
        return_document=ReturnDocument.AFTER,
    )
    return CommunityPostResponse(
        post_id=updated["post_id"], user_id=updated["user_id"],
        author_name=updated.get("author_name", ""), post_type=updated["post_type"],
        title=updated.get("title", ""), content=updated.get("content", ""),
        topic=updated.get("topic", ""), sub_topic=updated.get("sub_topic", ""),
        likes=updated.get("likes", []), dislikes=updated.get("dislikes", []),
        comments=[CommunityComment(**c) for c in updated.get("comments", [])],
        created_at=updated["created_at"],
    )


@app.delete("/community/posts/{post_id}/comments/{comment_id}", response_model=CommunityPostResponse, tags=["Community"])
async def delete_comment(post_id: str, comment_id: str, user_id: str = Query(...)):
    """Remove a comment (only the comment author can delete)."""
    db = await get_async_database()
    col = db[COLLECTIONS["community_posts"]]
    doc = await col.find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Post not found")

    comment = next((c for c in doc.get("comments", []) if c.get("comment_id") == comment_id), None)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    if comment.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your comment")

    updated = await col.find_one_and_update(
        {"post_id": post_id},
        {"$pull": {"comments": {"comment_id": comment_id}}},
        return_document=ReturnDocument.AFTER,
    )
    return CommunityPostResponse(
        post_id=updated["post_id"], user_id=updated["user_id"],
        author_name=updated.get("author_name", ""), post_type=updated["post_type"],
        title=updated.get("title", ""), content=updated.get("content", ""),
        topic=updated.get("topic", ""), sub_topic=updated.get("sub_topic", ""),
        likes=updated.get("likes", []), dislikes=updated.get("dislikes", []),
        comments=[CommunityComment(**c) for c in updated.get("comments", [])],
        created_at=updated["created_at"],
    )


@app.delete("/community/posts/{post_id}", tags=["Community"])
async def delete_community_post(post_id: str, user_id: str = Query(...)):
    """Delete a community post (only the post author can delete)."""
    db = await get_async_database()
    col = db[COLLECTIONS["community_posts"]]
    doc = await col.find_one({"post_id": post_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Post not found")
    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your post")
    await col.delete_one({"post_id": post_id})
    return {"ok": True, "deleted": post_id}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
