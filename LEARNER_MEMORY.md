# Learner memory (Pinecone)

Unified vectors for **MCQ generation** and **Tutor V2** so the same user gets retrieval-conditioned prompts on repeat **topic + sub_topic** lanes.

## Index

- Create a dedicated Pinecone index (do **not** reuse the Assignment Generator textbook index).
- **Dimensions: 1048** (required for the embedding model below).
- Env: `PINECONE_INDEX_LEARNER` = index name.

## Embedding

- Model: **`llama-text-embed-v2`** (Pinecone Inference API).
- Env override: `PINECONE_EMBED_MODEL` (default `llama-text-embed-v2`).
- Upserts: `input_type=passage`. Queries: `input_type=query`.

## Auth

- `PINECONE_API_KEY` in `.env` for `mcq_gen` and `AI_Study_bot`.

## Metadata (filterable)

| Field | Notes |
|--------|--------|
| `user_id` | string, required for queries |
| `topic_key`, `sub_topic_key` | normalized keys (see `learner_memory.normalize_key`) |
| `topic`, `sub_topic` | display strings |
| `prompt` | MCQ query text or tutor message |
| `source` | `mcq_session`, `tutor_v2_start`, `tutor_v2_lesson`, `tutor_v2_quiz`, … |
| `session_id` | mcq or tutor session id |
| `weak_concepts` | JSON string array (optional) |
| `last_score` | number (optional), quiz % |
| `excerpt` | short text for embedding |
| `created_at` | ISO string |

## When disabled

If `PINECONE_API_KEY` or `PINECONE_INDEX_LEARNER` is missing, all learner-memory calls no-op; generation behaves as before.
