"""
MongoDB Database Configuration and Connection

Handles connection to MongoDB Atlas and provides database access.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "mcq_generator")

# Synchronous client for non-async operations
_sync_client = None
_sync_db = None


def get_sync_database():
    """
    Get synchronous MongoDB database instance.
    Used for operations within the existing MCQ generator workflow.
    """
    global _sync_client, _sync_db
    
    if _sync_db is None:
        _sync_client = MongoClient(MONGODB_URI)
        _sync_db = _sync_client[MONGODB_DB_NAME]
    
    return _sync_db


def close_sync_database():
    """Close synchronous database connection"""
    global _sync_client
    if _sync_client:
        _sync_client.close()


# Async client for FastAPI endpoints
_async_client = None
_async_db = None


async def get_async_database():
    """
    Get asynchronous MongoDB database instance.
    Used for FastAPI endpoint queries and responses.
    """
    global _async_client, _async_db
    
    if _async_db is None:
        _async_client = AsyncIOMotorClient(MONGODB_URI)
        _async_db = _async_client[MONGODB_DB_NAME]
    
    return _async_db


async def close_async_database():
    """Close asynchronous database connection"""
    global _async_client
    if _async_client:
        _async_client.close()


async def ensure_mcq_indexes():
    """
    Ensure indexes used by API read/write paths exist.
    Safe to call repeatedly on startup.
    """
    db = await get_async_database()
    sessions_col = db[COLLECTIONS["mcq_sessions"]]
    mcqs_col = db[COLLECTIONS["mcqs"]]

    await sessions_col.create_index("session_id", unique=True)
    await sessions_col.create_index("user_id")
    await sessions_col.create_index([("subject", 1), ("chapter", 1), ("created_at", -1)])

    await mcqs_col.create_index([("session_id", 1), ("question_number", 1)], unique=True)
    await mcqs_col.create_index([("subject", 1), ("chapter", 1), ("created_at", -1)])
    await mcqs_col.create_index("user_id")

    dedup_col = db["flashcard_feedback_dedup"]
    await dedup_col.create_index([("user_id", 1), ("client_event_id", 1)], unique=True)

    chat_col = db[COLLECTIONS["chat_sessions"]]
    await chat_col.create_index("session_id", unique=True)
    await chat_col.create_index("user_id")
    await chat_col.create_index([("user_id", 1), ("updated_at", -1)])

    assign_col = db[COLLECTIONS["user_assignments"]]
    await assign_col.create_index("assignment_id", unique=True)
    await assign_col.create_index("user_id")
    await assign_col.create_index([("user_id", 1), ("created_at", -1)])

    videos_col = db[COLLECTIONS["user_videos"]]
    await videos_col.create_index("video_id", unique=True)
    await videos_col.create_index("user_id")
    await videos_col.create_index([("user_id", 1), ("created_at", -1)])

    community_col = db[COLLECTIONS["community_posts"]]
    await community_col.create_index("post_id", unique=True)
    await community_col.create_index("post_type")
    await community_col.create_index("topic")
    await community_col.create_index([("created_at", -1)])
    await community_col.create_index([("post_type", 1), ("created_at", -1)])
    await community_col.create_index(
        [("title", "text"), ("content", "text"), ("topic", "text"), ("sub_topic", "text")]
    )


# Collection names
COLLECTIONS = {
    "mcq_sessions": "mcq_sessions",      # Generation sessions metadata
    "mcqs": "mcqs",                       # Individual MCQs
    "concepts": "concepts",               # Extracted concepts
    "flashcard_feedback_dedup": "flashcard_feedback_dedup",
    "chat_sessions": "chat_sessions",     # MCP Hub chat conversations per user
    "user_assignments": "user_assignments",  # Saved assignment text per user
    "user_videos": "user_videos",         # Saved Manim video sessions per user
    "community_posts": "community_posts", # Globally shared community posts
}
