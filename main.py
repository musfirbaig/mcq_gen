"""
Main entry point for MCQ Generator

Provides CLI and programmatic interfaces for generating MCQs.
"""

import os
import argparse
from typing import Optional, Literal
from dotenv import load_dotenv

from graph import create_mcq_graph
from state import MCQGeneratorState
from nodes.assembler import export_mcqs_to_markdown


class MCQGenerator:
    """
    High-level interface for MCQ generation.
    """
    
    def __init__(
        self,
        llm_provider: Literal["anthropic", "openai", "gemini", "groq"] = "openai",
        model: str = "nvidia/nemotron-3-super-120b-a12b:free",
        batch_size: int = 15
    ):
        """
        Initialize MCQ Generator.
        
        Args:
            llm_provider: "anthropic", "openai", "gemini", or "groq"
            model: Model name (e.g., "openai/gpt-oss-120b", "claude-3-5-sonnet-20241022", "gpt-4")
            batch_size: Number of concepts to process per batch (10-15 recommended)
        """
        load_dotenv()

        self.llm_provider = llm_provider
        self.model = model
        self.batch_size = batch_size
        
        # Verify API keys
        if llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")
        if llm_provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment")
        if llm_provider == "groq" and not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment")
        
        # Create graph
        self.app = create_mcq_graph()
    
    def generate_from_file(
        self,
        input_path: str,
        input_type: Literal["chapter", "mcqs"] = "chapter",
        output_path: Optional[str] = None,
        include_explanations: bool = True
    ):
        """
        Generate MCQs from input file.
        
        Args:
            input_path: Path to chapter.md or existing MCQ file
            input_type: "chapter" or "mcqs"
            output_path: Where to save generated MCQs (optional)
            include_explanations: Include explanation section in output
        
        Returns:
            List of CompleteMCQ objects
        """
        print("\n" + "="*60)
        print("MCQ GENERATOR - Starting Generation")
        print("="*60)
        print(f"Input: {input_path} ({input_type})")
        print(f"LLM: {self.llm_provider} - {self.model}")
        print(f"Batch size: {self.batch_size}")
        print("="*60)
        
        # Initialize state
        initial_state = MCQGeneratorState(
            input_source=input_path,
            input_type=input_type,
            concepts_queue=[],
            processed_concept_ids=[],
            current_batch=[],
            batch_size=self.batch_size,
            generated_stems=[],
            validated_questions=[],
            questions_with_distractors=[],
            final_mcqs=[],
            needs_more_batches=False,
            validation_failures=[],
            config={
                "llm_provider": self.llm_provider,
                "model": self.model,
            },
            metrics={}
        )
        
        # Run graph
        print("\nExecuting LangGraph workflow...")
        result = self.app.invoke(initial_state)
        
        # Print final metrics
        print("\n" + "="*60)
        print("FINAL METRICS")
        print("="*60)
        metrics = result.get("metrics", {})
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Export if output path provided
        if output_path:
            export_mcqs_to_markdown(
                result["final_mcqs"],
                output_path,
                include_explanations=include_explanations,
                title="Generated MCQs: Integration"
            )
        
        return result["final_mcqs"]


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate high-quality MCQs for calculus/integration topics"
    )
    
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input file (chapter.md or existing MCQs)"
    )
    
    parser.add_argument(
        "--input-type",
        "-t",
        choices=["chapter", "mcqs"],
        default="chapter",
        help="Type of input: 'chapter' content or existing 'mcqs'"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output markdown file"
    )
    
    parser.add_argument(
        "--llm",
        choices=["anthropic", "openai", "gemini", "groq"],
        default="gemini",
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name (e.g., gemini-2.5-pro, claude-3-5-sonnet-20241022, gpt-4)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Number of concepts to process per batch (default: 15)"
    )
    
    parser.add_argument(
        "--no-explanations",
        action="store_true",
        help="Exclude explanation section from output"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = MCQGenerator(
        llm_provider=args.llm,
        model=args.model,
        batch_size=args.batch_size
    )
    
    # Generate MCQs
    mcqs = generator.generate_from_file(
        input_path=args.input,
        input_type=args.input_type,
        output_path=args.output,
        include_explanations=not args.no_explanations
    )
    
    print(f"\n✓ Successfully generated {len(mcqs)} MCQs!")
    print(f"✓ Output saved to: {args.output}")


if __name__ == "__main__":
    main()
