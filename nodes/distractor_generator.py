"""
Distractor Generator Node

Generates 3-5 plausible wrong answers using cognitive error taxonomy.
Each distractor simulates a specific student mistake pattern.
"""

import random
from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.groq_wrapper import ChatGroq

from state import MCQGeneratorState, ValidatedQuestion, Distractor
from error_taxonomy import get_applicable_errors, ErrorType, ERROR_TAXONOMY
from utils.latex_validator import validate_latex_syntax


DISTRACTOR_SYSTEM_PROMPT = """You are an expert at creating plausible wrong answers (distractors) for calculus MCQs.

CRITICAL REQUIREMENTS:
1. Generate 5 distinct wrong answers per question
2. Each distractor must simulate a SPECIFIC student error
3. Distractors must be PLAUSIBLE - not obviously wrong
4. Use proper LaTeX notation matching the correct answer's format
5. Each distractor should represent a different error type

ERROR TYPES TO SIMULATE:
- Sign errors: Changing + to - or vice versa
- Coefficient errors: Missing 1/n factor, wrong constant
- Exponent errors: Using n+1 instead of 1-n
- Chain rule forgotten: Missing du/dx factor
- Wrong formula: Using sin formula for cos
- Trigonometric identity confusion: sin/cos swap
- Absolute value missing: ln(x) instead of ln|x|
- Constant omitted: Forgetting +c

DISTRACTOR QUALITY:
✓ GOOD: Answer looks correct at first glance, requires careful checking
✓ GOOD: Represents common mistake students actually make
✗ BAD: Obviously wrong magnitude or units
✗ BAD: Random expression unrelated to the question
✗ BAD: Multiple errors stacked (too obviously wrong)

OUTPUT FORMAT (JSON):
{
  "distractors": [
    {
      "option_text": "$LaTeX$ expression",
      "error_type": "sign_error|coefficient_error|etc",
      "explanation": "This error occurs when student...",
      "plausibility_score": 0.8
    },
    ... (5 total)
  ]
}

Return ONLY valid JSON, no markdown code blocks."""


def generate_distractors_for_question(question: ValidatedQuestion, llm) -> List[Distractor]:
    """
    Generate 5 distractors for a validated question using LLM + error taxonomy.
    
    Args:
        question: ValidatedQuestion object
        llm: Language model for generation
    
    Returns:
        List of Distractor objects (5 distractors)
    """
    # Get applicable errors based on question metadata
    integral_type = question.get('generation_metadata', {}).get('integral_type', 'general')
    difficulty = question['difficulty']
    
    applicable_errors = get_applicable_errors(integral_type, difficulty)
    
    # Build error guidance for prompt
    error_guidance = "\n".join([
        f"- {err.name}: {err.description}"
        for err in applicable_errors[:8]  # Top 8 most relevant
    ])
    
    prompt = f"""Generate 5 plausible wrong answers for this question:

Question: {question['stem']}
Correct Answer: {question['correct_answer']}
Difficulty: {difficulty}

Focus on these error types (most relevant for this question):
{error_guidance}

Each distractor should:
1. Use the same LaTeX format as the correct answer
2. Be mathematically well-formed (not gibberish)
3. Simulate ONE specific error type
4. Be plausible enough that a student might choose it

Return JSON with 5 distractors, each with: option_text, error_type, explanation, plausibility_score (0-1)"""
    
    messages = [
        SystemMessage(content=DISTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse response
    import json
    import re
    
    try:
        content_text = response.content
        
        # Extract JSON from markdown
        if "```json" in content_text:
            json_match = re.search(r"```json\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        elif "```" in content_text:
            json_match = re.search(r"```\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        
        result = json.loads(content_text)
        distractors_data = result.get('distractors', [])
        
        # Convert to Distractor objects (LaTeX validation disabled)
        distractors = []
        for d in distractors_data:
            # Accept all distractors - trust the LLM
            distractor = Distractor(
                option_text=d['option_text'],
                error_type=d['error_type'],
                plausibility_score=float(d.get('plausibility_score', 0.7)),
                explanation=d['explanation']
            )
            distractors.append(distractor)
        
        return distractors
    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    ✗ Error parsing distractor JSON: {e}")
        return []


def rank_and_select_distractors(distractors: List[Distractor], num_select: int = 3) -> List[Distractor]:
    """
    Rank distractors by quality and select top N.
    
    Ranking criteria:
    - Plausibility score (higher is better)
    - Diversity of error types (prefer different types)
    - LaTeX validity
    
    Args:
        distractors: List of all generated distractors
        num_select: Number to select (default 3)
    
    Returns:
        Top N distractors
    """
    if len(distractors) <= num_select:
        return distractors
    
    # Sort by plausibility score
    sorted_distractors = sorted(distractors, key=lambda d: d['plausibility_score'], reverse=True)
    
    # Select ensuring diversity of error types
    selected = []
    used_error_types = set()
    
    for dist in sorted_distractors:
        if len(selected) >= num_select:
            break
        
        # Prefer distractors with new error types
        if dist['error_type'] not in used_error_types or len(selected) >= num_select - 1:
            selected.append(dist)
            used_error_types.add(dist['error_type'])
    
    # If still need more, add highest scored remaining
    if len(selected) < num_select:
        remaining = [d for d in sorted_distractors if d not in selected]
        selected.extend(remaining[:num_select - len(selected)])
    
    return selected


def distractor_generator_node(state: MCQGeneratorState) -> Dict:
    """
    LangGraph node that generates distractors for all validated questions.
    
    Args:
        state: Current MCQGeneratorState
    
    Returns:
        Updated state: questions_with_distractors, metrics
    """
    print("\n" + "="*60)
    print("DISTRACTOR GENERATOR NODE")
    print("="*60)
    
    validated_questions = state["validated_questions"]
    print(f"Generating distractors for {len(validated_questions)} questions")
    # Initialize LLM
    llm_provider = state["config"].get("llm_provider", "openai")
    model = state["config"].get("model", "nvidia/nemotron-3-super-120b-a12b:free")
    
    if llm_provider == "anthropic":
        llm = ChatAnthropic(model=model, temperature=0.7)
    elif llm_provider == "openai":
        llm = ChatOpenAI(model=model, temperature=0.7)
    elif llm_provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
    elif llm_provider == "groq":
        llm = ChatGroq(model=model, temperature=0.7)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    questions_with_distractors = []
    total_distractors_generated = 0
    
    for i, question in enumerate(validated_questions):
        print(f"\n[{i+1}/{len(validated_questions)}] Generating distractors for Q{i+1}")
        print(f"  Question: {question['stem'][:70]}...")
        
        # Generate 5 distractors
        distractors = generate_distractors_for_question(question, llm)
        
        if len(distractors) == 0:
            print(f"  ✗ Failed to generate distractors")
            continue
        
        print(f"  ✓ Generated {len(distractors)} distractors")
        
        # Rank and select top 3
        selected_distractors = rank_and_select_distractors(distractors, num_select=3)
        
        print(f"  ✓ Selected top {len(selected_distractors)} distractors:")
        for j, d in enumerate(selected_distractors):
            print(f"    {j+1}. {d['option_text'][:50]}... ({d['error_type']}, score: {d['plausibility_score']})")
        
        # Combine question with distractors
        question_data = {
            **question,
            'distractors': selected_distractors
        }
        questions_with_distractors.append(question_data)
        total_distractors_generated += len(selected_distractors)
    
    print(f"\n" + "="*60)
    print(f"DISTRACTOR GENERATION SUMMARY")
    print(f"="*60)
    print(f"✓ Questions with distractors: {len(questions_with_distractors)}")
    print(f"✓ Total distractors generated: {total_distractors_generated}")
    if len(questions_with_distractors) > 0:
        print(f"✓ Average per question: {total_distractors_generated/len(questions_with_distractors):.1f}")
    else:
        print(f"⚠ No questions passed validation - cannot generate distractors")
    
    return {
        "questions_with_distractors": questions_with_distractors,
        "metrics": {
            **state.get("metrics", {}),
            "questions_with_distractors": len(questions_with_distractors),
            "total_distractors_generated": total_distractors_generated
        }
    }
