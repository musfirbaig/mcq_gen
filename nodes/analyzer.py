"""
Content Analyzer Node

Extracts structured concepts from chapter content or existing MCQs,
creating JSON objects for batch processing by the stem generator.

Handles two input modes:
1. Chapter content: Parses markdown to extract concepts, formulas, examples
2. Existing MCQs: Analyzes patterns to generate similar concept structures
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.groq_wrapper import ChatGroq

from state import MCQGeneratorState, ConceptJSON


ANALYZER_SYSTEM_PROMPT = """You are an expert mathematics educator analyzing calculus content to extract structured concepts for MCQ generation.

Your task is to parse the provided content and extract individual mathematical concepts, each as a JSON object.

For CHAPTER CONTENT:
- Identify distinct integration techniques, formulas, or theorems
- Extract worked examples as references
- Classify difficulty based on prerequisite knowledge

For EXISTING MCQs:
- Identify the underlying concept being tested
- Extract the formula or technique involved
- Infer difficulty from question complexity

Each concept should be ATOMIC - testable as a single MCQ. Don't create overly broad concepts.

Return a JSON array of concept objects with this exact structure:
{
  "concept_id": "unique_identifier_snake_case",
  "concept_name": "Human readable name",
  "formula": "LaTeX formula (if applicable)",
  "difficulty": "easy|medium|hard",
  "prerequisites": ["list", "of", "required", "concepts"],
  "context": "2-3 sentences explaining the concept and when to use it",
  "worked_example": "Optional: A worked problem demonstrating the concept"
}

CRITICAL RULES:
1. Extract 30-50 concepts for a full chapter
2. Each concept must be independently testable
3. Use proper LaTeX notation
4. Difficulty: easy (recall), medium (application), hard (multi-step/synthesis)
5. Return ONLY valid JSON, no markdown formatting
"""

def _prepare_content_for_prompt(content: str) -> str:
    """
    Keep analyzer payload under provider limits to avoid timeout/size errors.
    """
    max_chars = int(os.getenv("ANALYZER_MAX_CHARS", "12000"))
    if len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + "\n\n[TRUNCATED CONTENT]\n"
        + "Content was truncated to stay within token limits."
    )


def analyze_chapter_content(content: str, llm) -> List[ConceptJSON]:
    """
    Parse chapter markdown to extract mathematical concepts.
    
    Args:
        content: Raw markdown text from chapter file
        llm: Language model for extraction
    
    Returns:
        List of ConceptJSON objects
    """
    safe_content = _prepare_content_for_prompt(content)
    messages = [
        SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Extract mathematical concepts from this chapter content:

{safe_content}

Return a JSON array of concept objects. Aim for 40-50 concepts covering all major topics.""")
    ]
    
    response = llm.invoke(messages)
    
    # Parse JSON response
    try:
        # Try to extract JSON from markdown code blocks if present
        content_text = response.content
        if "```json" in content_text:
            json_match = re.search(r"```json\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        elif "```" in content_text:
            json_match = re.search(r"```\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        
        # Use json.loads with strict=False to be more lenient with escape sequences
        concepts = json.loads(content_text, strict=False)
        
        # Validate structure
        validated_concepts = []
        for i, concept in enumerate(concepts):
            try:
                # Ensure all required fields exist
                validated = ConceptJSON(
                    concept_id=concept.get("concept_id", f"concept_{i+1}"),
                    concept_name=concept["concept_name"],
                    formula=concept.get("formula", ""),
                    difficulty=concept.get("difficulty", "medium"),
                    prerequisites=concept.get("prerequisites", []),
                    context=concept["context"],
                    worked_example=concept.get("worked_example")
                )
                validated_concepts.append(validated)
            except KeyError as e:
                print(f"Warning: Skipping malformed concept {i+1}: {e}")
                continue
        
        return validated_concepts
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM response: {e}")
        print(f"Response preview (first 500 chars): {response.content[:500]}")
        # Save full response for debugging
        debug_path = "analyzer_error_response.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(response.content)
        print(f"Full response saved to: {debug_path}")
        return []


def analyze_existing_mcqs(content: str, llm) -> List[ConceptJSON]:
    """
    Parse existing MCQs to extract concepts for generating similar questions.
    
    Args:
        content: Raw markdown text from MCQ file
        llm: Language model for extraction
    
    Returns:
        List of ConceptJSON objects inferred from MCQ patterns
    """
    safe_content = _prepare_content_for_prompt(content)
    messages = [
        SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze these existing MCQs and extract the underlying mathematical concepts:

{safe_content}

For each unique concept/technique being tested, create a concept object. 
Return a JSON array of 30-40 concepts that could generate similar questions.""")
    ]
    
    response = llm.invoke(messages)
    
    # Parse JSON response (same logic as analyze_chapter_content)
    try:
        content_text = response.content
        if "```json" in content_text:
            json_match = re.search(r"```json\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        elif "```" in content_text:
            json_match = re.search(r"```\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        
        # Use json.loads with strict=False to be more lenient with escape sequences
        concepts = json.loads(content_text, strict=False)
        
        validated_concepts = []
        for i, concept in enumerate(concepts):
            try:
                validated = ConceptJSON(
                    concept_id=concept.get("concept_id", f"mcq_concept_{i+1}"),
                    concept_name=concept["concept_name"],
                    formula=concept.get("formula", ""),
                    difficulty=concept.get("difficulty", "medium"),
                    prerequisites=concept.get("prerequisites", []),
                    context=concept["context"],
                    worked_example=concept.get("worked_example")
                )
                validated_concepts.append(validated)
            except KeyError as e:
                print(f"Warning: Skipping malformed concept {i+1}: {e}")
                continue
        
        return validated_concepts
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM response: {e}")
        print(f"Response preview (first 500 chars): {response.content[:500]}")
        # Save full response for debugging
        debug_path = "analyzer_error_response.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(response.content)
        print(f"Full response saved to: {debug_path}")
        return []


def save_analyzer_intermediate_data(concepts: List[ConceptJSON], state: MCQGeneratorState):
    """
    Save analyzer intermediate data for debugging and understanding.
    
    Args:
        concepts: List of extracted concepts
        state: Current MCQGeneratorState
    """
    # Create intermediate_data directory if it doesn't exist
    output_dir = state.get("output_dir", "intermediate_data")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save concepts as JSON
    concepts_json_path = os.path.join(output_dir, "01_concepts.json")
    with open(concepts_json_path, "w", encoding="utf-8") as f:
        json.dump([dict(c) for c in concepts], f, indent=2, ensure_ascii=False)
    
    # Save concepts in readable format
    concepts_readable_path = os.path.join(output_dir, "01_concepts_readable.txt")
    with open(concepts_readable_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CONTENT ANALYZER - EXTRACTED CONCEPTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input Source: {state['input_source']}\n")
        f.write(f"Input Type: {state['input_type']}\n")
        f.write(f"LLM Provider: {state['config'].get('llm_provider', 'gemini')}\n")
        f.write(f"Model: {state['config'].get('model', 'gemini-2.5-pro')}\n")
        f.write(f"Total Concepts Extracted: {len(concepts)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Group concepts by difficulty
        difficulty_groups = {"easy": [], "medium": [], "hard": []}
        for concept in concepts:
            diff = concept.get("difficulty", "medium")
            if diff in difficulty_groups:
                difficulty_groups[diff].append(concept)
        
        f.write("DIFFICULTY DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        for difficulty, group in difficulty_groups.items():
            f.write(f"  {difficulty.upper()}: {len(group)} concepts\n")
        f.write("\n")
        
        # Write detailed concept information
        for i, concept in enumerate(concepts, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CONCEPT {i}: {concept['concept_name']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"ID: {concept['concept_id']}\n")
            f.write(f"Difficulty: {concept.get('difficulty', 'medium')}\n")
            
            if concept.get('prerequisites'):
                f.write(f"Prerequisites: {', '.join(concept['prerequisites'])}\n")
            
            f.write(f"\nContext:\n{concept['context']}\n")
            
            if concept.get('formula'):
                f.write(f"\nFormula:\n{concept['formula']}\n")
            
            if concept.get('worked_example'):
                f.write(f"\nWorked Example:\n{concept['worked_example']}\n")
            
            f.write("\n")
    
    print(f"\nSaved intermediate data:")
    print(f"  - {concepts_json_path}")
    print(f"  - {concepts_readable_path}")


def content_analyzer_node(state: MCQGeneratorState) -> Dict:
    """
    LangGraph node that analyzes input content and populates concepts queue.
    
    Args:
        state: Current MCQGeneratorState
    
    Returns:
        Updated state fields: concepts_queue, current_batch, needs_more_batches
    """
    print("\n" + "="*60)
    print("CONTENT ANALYZER NODE")
    print("="*60)
    # Initialize LLM based on config
    llm_provider = state["config"].get("llm_provider", "openai")
    model = state["config"].get("model", "nvidia/nemotron-3-super-120b-a12b:free")
    
    if llm_provider == "anthropic":
        llm = ChatAnthropic(model=model, temperature=0.3)
    elif llm_provider == "openai":
        llm = ChatOpenAI(model=model, temperature=0.3)
    elif llm_provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
    elif llm_provider == "groq":
        llm = ChatGroq(model=model, temperature=0.3)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    # Read input file
    with open(state["input_source"], "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract concepts based on input type
    if state["input_type"] == "chapter":
        print(f"Analyzing chapter content from: {state['input_source']}")
        concepts = analyze_chapter_content(content, llm)
    else:  # mcqs
        print(f"Analyzing existing MCQs from: {state['input_source']}")
        concepts = analyze_existing_mcqs(content, llm)
    
    print(f"Extracted {len(concepts)} concepts")
    
    # Log first few concepts for debugging
    for i, concept in enumerate(concepts[:3]):
        print(f"\nConcept {i+1}: {concept['concept_name']} ({concept['difficulty']})")
        print(f"  Formula: {concept['formula'][:60]}..." if len(concept['formula']) > 60 else f"  Formula: {concept['formula']}")
    
    # Save intermediate data
    save_analyzer_intermediate_data(concepts, state)
    
    # Prepare batching
    batch_size = state.get("batch_size", 15)
    current_batch = concepts[:batch_size]
    remaining_concepts = concepts[batch_size:]
    
    print(f"\nPrepared first batch of {len(current_batch)} concepts")
    print(f"Remaining in queue: {len(remaining_concepts)}")
    
    return {
        "concepts_queue": remaining_concepts,
        "current_batch": current_batch,
        "needs_more_batches": len(remaining_concepts) > 0,
        "metrics": {
            **state.get("metrics", {}),
            "total_concepts_extracted": len(concepts),
            "concepts_in_first_batch": len(current_batch)
        }
    }
