"""
FastAPI Server for MCQ Generator

Provides REST API endpoints for generating MCQs and storing them in MongoDB Atlas.
Wraps the existing MCQGenerator without modifying its internal logic.
"""

import os
import uuid
import tempfile
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from main import MCQGenerator
from storage import MCQStorageService
from database import get_async_database, close_async_database, COLLECTIONS
from models import (
    GenerateMCQResponse, MCQResponse, SessionResponse,
    MCQListResponse, SessionListResponse, HealthResponse
)
from nodes.assembler import export_mcqs_to_markdown

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
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-pro")
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "15"))


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    await get_async_database()
    print("✓ FastAPI server started")
    print("✓ MongoDB connection initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    await close_async_database()
    print("✓ MongoDB connection closed")


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
            "health": "GET /health"
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
    file: UploadFile = File(..., description="Input file (chapter.md or existing MCQs)"),
    subject: str = Form(..., description="Subject name (e.g., 'Calculus', 'Linear Algebra')"),
    chapter: str = Form(..., description="Chapter name (e.g., 'Chapter 3 - Definite Integrals')"),
    input_type: str = Form("chapter", description="Type of input: 'chapter' or 'mcqs'"),
    include_explanations: bool = Form(True, description="Include explanations in MCQs")
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
    
    # Validate input_type
    if input_type not in ["chapter", "mcqs"]:
        raise HTTPException(status_code=400, detail="input_type must be 'chapter' or 'mcqs'")
    
    # Validate file type
    if not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="File must be a markdown (.md) file")
    
    # Use configuration from environment variables
    llm_provider = DEFAULT_LLM_PROVIDER
    model = DEFAULT_MODEL
    batch_size = DEFAULT_BATCH_SIZE
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Initialize storage service
    storage = MCQStorageService(session_id=session_id)
    
    # Create temporary file to store uploaded content
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.md') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Create session record in database
        storage.save_session(
            subject=subject,
            chapter=chapter,
            input_filename=file.filename,
            input_type=input_type,
            llm_provider=llm_provider,
            model=model,
            batch_size=batch_size,
            status="processing"
        )
        
        print(f"\n{'='*60}")
        print(f"API REQUEST - Session ID: {session_id}")
        print(f"{'='*60}")
        print(f"Subject: {subject}")
        print(f"Chapter: {chapter}")
        print(f"File: {file.filename}")
        print(f"Input Type: {input_type}")
        print(f"LLM: {llm_provider} - {model}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Initialize MCQ Generator with specified configuration
        generator = MCQGenerator(
            llm_provider=llm_provider,
            model=model,
            batch_size=batch_size
        )
        
        # Generate MCQs (synchronous - waits for completion)
        mcqs = generator.generate_from_file(
            input_path=temp_file_path,
            input_type=input_type,
            output_path=None,  # We'll handle export separately
            include_explanations=include_explanations
        )
        
        # Get the final state to extract concepts and metrics
        # We need to re-run just the analyzer to get concepts for storage
        from state import MCQGeneratorState
        from nodes.analyzer import content_analyzer_node
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI
        
        # Read file content for concept extraction
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Create a minimal state to extract concepts
        temp_state = MCQGeneratorState(
            input_source=temp_file_path,
            input_type=input_type,
            concepts_queue=[],
            processed_concept_ids=[],
            current_batch=[],
            batch_size=batch_size,
            generated_stems=[],
            validated_questions=[],
            questions_with_distractors=[],
            final_mcqs=[],
            needs_more_batches=False,
            validation_failures=[],
            config={
                "llm_provider": llm_provider,
                "model": model,
            },
            metrics={}
        )
        
        # Extract concepts
        analyzer_result = content_analyzer_node(temp_state)
        concepts = analyzer_result.get("current_batch", []) + analyzer_result.get("concepts_queue", [])
        
        # Save concepts to database, chapter=chapter
        if concepts:
            storage.save_concepts(concepts, subject=subject, chapter=chapter)
        
        # Save MCQs to database
        storage.save_mcqs(mcqs, subject=subject, chapter=chapter)
        
        # Calculate metrics
        difficulty_dist = {}
        for mcq in mcqs:
            diff = mcq['metadata']['difficulty']
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        
        metrics = {
            "total_concepts_extracted": len(concepts),
            "total_mcqs_generated": len(mcqs),
            "validation_rate": 1.0,  # Since validator is pass-through
            "difficulty_distribution": difficulty_dist
        }
        
        # Update session with completion status
        storage.update_session(
            total_concepts=len(concepts),
            total_mcqs=len(mcqs),
            difficulty_dist=difficulty_dist,
            metrics=metrics,
            status="completed"
        )
        
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
        
        # Convert MCQs to response format
        mcq_responses = []
        db = await get_async_database()
        
        # Fetch saved MCQs from database to get their IDs
        saved_mcqs = await db[COLLECTIONS["mcqs"]].find(
            {"session_id": session_id}
        ).to_list(length=None)
        
        for saved_mcq in saved_mcqs:
            mcq_responses.append(MCQResponse(
                id=str(saved_mcq["_id"]),
                session_id=saved_mcq["session_id"],
                subject=saved_mcq["subject"],
                chapter=saved_mcq["chapter"],
                question_number=saved_mcq["question_number"],
                concept_id=saved_mcq["concept_id"],
                stem=saved_mcq["stem"],
                options=saved_mcq["options"],
                correct_answer=saved_mcq["correct_answer"],
                explanation=saved_mcq["explanation"],
                metadata=saved_mcq["metadata"],
                created_at=saved_mcq["created_at"]
            ))
        
        print(f"\n{'='*60}")
        print(f"✓ Generation completed successfully!")
        print(f"✓ Session ID: {session_id}")
        print(f"✓ MCQs generated: {len(mcqs)}")
        print(f"✓ Saved to MongoDB Atlas")
        print(f"{'='*60}\n")
        
        return GenerateMCQResponse(
            session_id=session_id,
            message="MCQs generated successfully",
            total_mcqs_generated=len(mcqs),
            difficulty_distribution=difficulty_dist,
            metrics=metrics,
            mcqs=mcq_responses,
            markdown_content=markdown_content
        )
    
    except Exception as e:
        # Update session with error status
        storage.update_session(
            status="failed",
            error_message=str(e)
        )
        
        print(f"\n{'='*60}")
        print(f"✗ Generation failed!")
        print(f"✗ Session ID: {session_id}")
        print(f"✗ Error: {str(e)}")
        print(f"{'='*60}\n")
        
        raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(
    subject: Optional[str] = Query(None, description="Filter by subject"),
    chapter: Optional[str] = Query(None, description="Filter by chapter"),
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum sessions to return")
):
    """
    List all MCQ generation sessions.
    
    Optionally filter by subject and/or chapter. Returns paginated list of sessions with metadata.
    """
    db = await get_async_database()
    
    # Build query filter
    query_filter = {}
    if subject:
        query_filter["subject"] = subject
    if chapter:
        query_filter["chapter"] = chapter
    
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
            subject=session["subject"],
            chapter=session["chapter"],
            input_filename=session["input_filename"],
            input_type=session["input_type"],
            llm_provider=session["llm_provider"],
            model=session["model"],
            total_concepts_extracted=session["total_concepts_extracted"],
            total_mcqs_generated=session["total_mcqs_generated"],
            difficulty_distribution=session["difficulty_distribution"],
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
        subject=session["subject"],
        chapter=session["chapter"],
        input_filename=session["input_filename"],
        input_type=session["input_type"],
        llm_provider=session["llm_provider"],
        model=session["model"],
        total_concepts_extracted=session["total_concepts_extracted"],
        total_mcqs_generated=session["total_mcqs_generated"],
        difficulty_distribution=session["difficulty_distribution"],
        status=session["status"],
        created_at=session["created_at"],
        completed_at=session.get("completed_at")
    )


@app.get("/mcqs", response_model=MCQListResponse, tags=["MCQs"])
async def list_mcqs(
    subject: Optional[str] = Query(None, description="Filter by subject"),
    chapter: Optional[str] = Query(None, description="Filter by chapter"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (easy, medium, hard)"),
    skip: int = Query(0, ge=0, description="Number of MCQs to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum MCQs to return")
):
    """
    List all generated MCQs.
    
    Optionally filter by subject, chapter, session ID, and/or difficulty.
    """
    db = await get_async_database()
    
    # Build query filter
    query_filter = {}
    if subject:
        query_filter["subject"] = subject
    if chapter:
        query_filter["chapter"] = chapter
    if session_id:
        query_filter["session_id"] = session_id
    if difficulty:
        query_filter["metadata.difficulty"] = difficulty
    
    # Get total count
    total = await db[COLLECTIONS["mcqs"]].count_documents(query_filter)
    
    # Fetch MCQs
    mcqs = await db[COLLECTIONS["mcqs"]].find(query_filter)\
        .sort("question_number", 1)\
        .skip(skip)\
        .limit(limit)\
        .to_list(length=limit)
    
    mcq_responses = []
    for mcq in mcqs:
        mcq_responses.append(MCQResponse(
            id=str(mcq["_id"]),
            session_id=mcq["session_id"],
            subject=mcq["subject"],
            chapter=mcq["chapter"],
            question_number=mcq["question_number"],
            concept_id=mcq["concept_id"],
            stem=mcq["stem"],
            options=mcq["options"],
            correct_answer=mcq["correct_answer"],
            explanation=mcq["explanation"],
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
        id=str(mcq["_id"]),
        session_id=mcq["session_id"],
        subject=mcq["subject"],
        chapter=mcq["chapter"],
        question_number=mcq["question_number"],
        concept_id=mcq["concept_id"],
        stem=mcq["stem"],
        options=mcq["options"],
        correct_answer=mcq["correct_answer"],
        explanation=mcq["explanation"],
        metadata=mcq["metadata"],
        created_at=mcq["created_at"]
    )


# --- Vercel serverless handler ---
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # mangum not needed when running locally with uvicorn


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
