"""
Stem Generator Node

Generates question stems with correct answers from concept JSON objects.
Processes batches of 10-15 concepts to prevent hallucination through focused prompting.

Includes loopback logic: after processing current batch, loads next batch if queue not empty.
"""

import uuid
from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.groq_wrapper import ChatGroq

from state import MCQGeneratorState, ConceptJSON, StemWithAnswer
from utils.latex_validator import validate_latex_syntax, extract_latex_from_markdown


STEM_GENERATOR_SYSTEM_PROMPT = """You are an expert mathematics educator creating high-quality MCQ questions for calculus/integration.

CRITICAL REQUIREMENTS:
1. Generate ONE question with correct answer per concept
2. Question must directly test the specific concept provided
3. Use proper LaTeX notation in $...$ for math expressions
4. Correct answer must be mathematically accurate
5. Question difficulty should match the concept's difficulty level

QUESTION TYPES by Difficulty:
- EASY: Direct formula application, recall
  Example: "What is ∫x^n dx = ?"
  
- MEDIUM: One-step problem solving, substitution
  Example: "∫(2x+3)^5 dx = ?"
  
- HARD: Multi-step, integration by parts, complex substitutions
  Example: "∫x²e^(x³)cos(e^(x³)) dx = ?"

OUTPUT FORMAT (JSON):
{
  "stem": "Question text with $LaTeX$ expressions",
  "correct_answer": "$LaTeX$ expression for the correct answer",
  "integral_type": "substitution|by_parts|trigonometric|exponential|etc",
  "reasoning": "Brief explanation of why this answer is correct"
}

LATEX RULES:
- Use \\frac{a}{b} for fractions
- Use \\int for integral symbol, \\int_a^b for definite integrals
- Use ^{} for superscripts, _{} for subscripts
- Use \\sin, \\cos, \\tan (not sin, cos, tan)
- Use \\sin^{-1} for inverse trig (not \\arcsin)
- Use \\ln|x| with absolute value bars where appropriate
- Always include +c for indefinite integrals

Return ONLY valid JSON, no markdown code blocks."""


def generate_stem_for_concept(concept: ConceptJSON, llm) -> Dict:
    """
    Generate a single question stem with correct answer for one concept.
    
    Args:
        concept: ConceptJSON object with concept details
        llm: Language model for generation
    
    Returns:
        Dict with stem, correct_answer, integral_type, reasoning
    """
    prompt = f"""Generate ONE MCQ question for this concept:

Concept: {concept['concept_name']}
Formula: {concept['formula']}
Difficulty: {concept['difficulty']}
Context: {concept['context']}
{f"Example: {concept.get('worked_example', '')}" if concept.get('worked_example') else ""}

Create a {concept['difficulty']}-level question that tests understanding of this concept.
The question should be clear, unambiguous, and have one correct answer.

Return JSON with: stem, correct_answer, integral_type, reasoning"""
    
    messages = [
        SystemMessage(content=STEM_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse response
    import json
    import re
    
    try:
        content_text = response.content
        
        # Extract JSON from markdown code blocks if present
        if "```json" in content_text:
            json_match = re.search(r"```json\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        elif "```" in content_text:
            json_match = re.search(r"```\s*(.*?)\s*```", content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
        
        result = json.loads(content_text)
        return result
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for concept {concept['concept_id']}: {e}")
        print(f"Response was: {response.content[:300]}")
        return None


def stem_generator_node(state: MCQGeneratorState) -> Dict:
    """
    LangGraph node that generates stems for current batch of concepts.
    Implements loopback logic: loads next batch if concepts_queue not empty.
    
    Args:
        state: Current MCQGeneratorState
    
    Returns:
        Updated state: generated_stems, current_batch, needs_more_batches, processed_concept_ids
    """
    print("\n" + "="*60)
    print("STEM GENERATOR NODE")
    print("="*60)
    
    current_batch = state["current_batch"]
    print(f"Processing batch of {len(current_batch)} concepts")
    # Initialize LLM
    llm_provider = state["config"].get("llm_provider", "openai")
    model = state["config"].get("model", "nvidia/nemotron-3-super-120b-a12b:free")
    
    if llm_provider == "anthropic":
        llm = ChatAnthropic(model=model, temperature=0.5)
    elif llm_provider == "openai":
        llm = ChatOpenAI(model=model, temperature=0.5)
    elif llm_provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.5)
    elif llm_provider == "groq":
        llm = ChatGroq(model=model, temperature=0.5)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    # Generate stems for each concept in current batch
    new_stems = []
    processed_ids = state.get("processed_concept_ids", [])
    
    for i, concept in enumerate(current_batch):
        print(f"\n[{i+1}/{len(current_batch)}] Generating question for: {concept['concept_name']}")
        
        result = generate_stem_for_concept(concept, llm)
        
        if result is None:
            print(f"  ⚠ Failed to generate stem for {concept['concept_id']}")
            continue
        
        # Validate LaTeX in stem and answer
        stem_latex = extract_latex_from_markdown(result['stem'])
        answer_latex = extract_latex_from_markdown(result['correct_answer'])
        
        latex_valid = True
        for latex_expr in stem_latex + answer_latex:
            is_valid, error = validate_latex_syntax(latex_expr)
            if not is_valid:
                print(f"  ⚠ LaTeX validation failed: {error}")
                latex_valid = False
                break
        
        # Create StemWithAnswer object
        stem_obj = StemWithAnswer(
            question_id=str(uuid.uuid4()),
            concept_id=concept['concept_id'],
            stem=result['stem'],
            correct_answer=result['correct_answer'],
            difficulty=concept['difficulty'],
            latex_valid=latex_valid,
            generation_metadata={
                'integral_type': result.get('integral_type', 'unknown'),
                'reasoning': result.get('reasoning', '')
            }
        )
        
        new_stems.append(stem_obj)
        processed_ids.append(concept['concept_id'])
        
        print(f"  ✓ Generated: {result['stem'][:80]}...")
        print(f"  ✓ Answer: {result['correct_answer']}")
        print(f"  ✓ LaTeX valid: {latex_valid}")
    
    # Merge with existing stems
    all_stems = state.get("generated_stems", []) + new_stems
    
    print(f"\n✓ Generated {len(new_stems)} stems for this batch")
    print(f"✓ Total stems generated so far: {len(all_stems)}")
    
    # LOOPBACK LOGIC: Load next batch if queue not empty
    concepts_queue = state["concepts_queue"]
    batch_size = state.get("batch_size", 15)
    
    # Check if there are more concepts to process
    if len(concepts_queue) > 0:
        next_batch = concepts_queue[:batch_size]
        remaining_queue = concepts_queue[batch_size:]
        
        # needs_more_batches should be True if we have a next_batch to process
        # It becomes False only when concepts_queue is empty
        needs_more = True
        
        print(f"\n→ Loading next batch of {len(next_batch)} concepts")
        print(f"→ {len(remaining_queue)} concepts remaining in queue")
        
        return {
            "generated_stems": all_stems,
            "current_batch": next_batch,
            "concepts_queue": remaining_queue,
            "needs_more_batches": needs_more,
            "processed_concept_ids": processed_ids,
            "metrics": {
                **state.get("metrics", {}),
                "stems_generated": len(all_stems),
                "stems_in_last_batch": len(new_stems)
            }
        }
    else:
        print(f"\n✓ All concepts processed! Total stems: {len(all_stems)}")
        
        return {
            "generated_stems": all_stems,
            "current_batch": [],
            "concepts_queue": [],
            "needs_more_batches": False,
            "processed_concept_ids": processed_ids,
            "metrics": {
                **state.get("metrics", {}),
                "stems_generated": len(all_stems),
                "stems_in_last_batch": len(new_stems)
            }
        }
