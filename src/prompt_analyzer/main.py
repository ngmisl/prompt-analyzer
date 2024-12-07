import os
from pathlib import Path
from typing import Dict, List
import asyncio
from concurrent.futures import TimeoutError
import time

from dotenv import load_dotenv, set_key
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


class LLMConfig(BaseModel):
    name: str
    model_id: str
    description: str


class PromptAnalysis(BaseModel):
    initial_analysis: str
    rewritten_prompt: str
    potential_improvements: List[str]
    refined_prompt: str
    final_prompt: str
    models_used: Dict[str, str]


class PromptAnalyzer:
    DEFAULT_MODELS = {
        "llama": LLMConfig(
            name="Llama 3",
            model_id="meta-llama/llama-3.3-70b-instruct",
            description="High performance instruct model",
        ),
        "nova": LLMConfig(
            name="Nova Lite",
            model_id="amazon/nova-lite-v1",
            description="Efficient general purpose model",
        ),
        "learnlm": LLMConfig(
            name="LearnLM Pro",
            model_id="google/learnlm-1.5-pro-experimental:free",
            description="Specialized learning model",
        ),
    }

    DEFAULT_PIPELINE = {
        "analysis": "llama",
        "rewrite": "nova",
        "improvements": "learnlm",
        "refinement": "llama",
        "final": "nova",
    }

    def __init__(self):
        self.setup_environment()
        self.models = {}
        self.pipeline = self.DEFAULT_PIPELINE
        self.timeout = 30  # timeout in seconds
        self.initialize_models()

    def setup_environment(self):
        env_path = Path(".env")
        if not env_path.exists():
            api_key = input("Enter your OpenRouter API key: ")
            with open(env_path, "w") as f:
                f.write(f"OPENROUTER_API_KEY={api_key}")
            os.environ["OPENROUTER_API_KEY"] = api_key
        else:
            load_dotenv(env_path)
            if not os.getenv("OPENROUTER_API_KEY"):
                api_key = input("OpenRouter API key not found. Please enter it: ")
                set_key(env_path, "OPENROUTER_API_KEY", api_key)
                os.environ["OPENROUTER_API_KEY"] = api_key

    def initialize_models(self):
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API Key is not set")

        try:
            for model_key, config in self.DEFAULT_MODELS.items():
                self.models[model_key] = ChatOpenAI(
                    model=config.model_id,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=SecretStr(openrouter_api_key),
                    temperature=0.7,
                    timeout=self.timeout,  # Changed from request_timeout to timeout
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize models: {str(e)}") from e

    def set_pipeline(self, custom_pipeline: Dict[str, str]):
        """Set custom pipeline if needed"""
        self.pipeline = custom_pipeline

    async def run_with_timeout(self, chain, inputs: dict, step_name: str):
        try:
            print(f"\nExecuting {step_name}...")
            start_time = time.time()
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, chain.invoke, inputs),
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time
            print(f"✓ {step_name} completed in {elapsed_time:.2f}s")
            return result
        except TimeoutError:
            raise TimeoutError(f"Operation timed out while executing {step_name}")
        except Exception as e:
            raise Exception(f"Error in {step_name}: {str(e)}")

    async def analyze(self, prompt: str) -> PromptAnalysis:
        # Analysis
        analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze this prompt's structure and effectiveness:\n\n{prompt}"
        )
        analysis_chain = analysis_prompt | self.models[self.pipeline["analysis"]]
        initial_analysis = await self.run_with_timeout(
            analysis_chain, {"prompt": prompt}, "Initial Analysis"
        )

        # Rewrite
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this prompt for clarity:\n\n{prompt}\n\nAnalysis: {analysis}"
        )
        rewrite_chain = rewrite_prompt | self.models[self.pipeline["rewrite"]]
        rewritten = await self.run_with_timeout(
            rewrite_chain, {"prompt": prompt, "analysis": initial_analysis.content}, "Rewriting"
        )

        # Improvements
        improve_prompt = ChatPromptTemplate.from_template(
            "List specific improvements for this prompt:\n\n{prompt}"
        )
        improve_chain = improve_prompt | self.models[self.pipeline["improvements"]]
        improvements = await self.run_with_timeout(
            improve_chain, {"prompt": rewritten.content}, "Generating Improvements"
        )

        # Refinement
        refine_prompt = ChatPromptTemplate.from_template(
            "Refine this prompt:\n\n{prompt}\n\nSuggested improvements: {improvements}"
        )
        refine_chain = refine_prompt | self.models[self.pipeline["refinement"]]
        refined = await self.run_with_timeout(
            refine_chain,
            {"prompt": rewritten.content, "improvements": improvements.content},
            "Refining",
        )

        # Final
        final_prompt = ChatPromptTemplate.from_template(
            "Improve this prompt concisely, outputting only the improved version with no explanation:\n\n{prompt}"
        )
        final_chain = final_prompt | self.models[self.pipeline["final"]]
        final = await self.run_with_timeout(final_chain, {"prompt": refined.content}, "Finalizing")

        # Process improvements list
        improvement_list = [
            imp.strip()
            for imp in improvements.content.split("\n")
            if imp.strip() and not imp.strip().isspace()
        ]
        improvement_list = [imp.lstrip("•-*[] 1234567890.").strip() for imp in improvement_list]

        return PromptAnalysis(
            initial_analysis=initial_analysis.content,
            rewritten_prompt=rewritten.content,
            potential_improvements=improvement_list,
            refined_prompt=refined.content,
            final_prompt=final.content,
            models_used={
                stage: self.DEFAULT_MODELS[model_key].name
                for stage, model_key in self.pipeline.items()
            },
        )


async def main():
    analyzer = PromptAnalyzer()

    print("\nPrompt Analysis and Refinement Tool")
    print("===================================")

    # Show available models
    print("\nAvailable Models:")
    for key, model in analyzer.DEFAULT_MODELS.items():
        print(f"- {model.name} ({key}): {model.description}")

    # Optional custom pipeline
    use_custom = input("\nCustomize model pipeline? (y/n): ").lower() == "y"
    if use_custom:
        custom_pipeline = {}
        stages = ["analysis", "rewrite", "improvements", "refinement", "final"]
        print("\nEnter model key for each stage:")
        for stage in stages:
            while True:
                model_key = input(f"{stage}: ")
                if model_key in analyzer.DEFAULT_MODELS:
                    custom_pipeline[stage] = model_key
                    break
                print("Invalid model key. Try again.")
        analyzer.set_pipeline(custom_pipeline)

    prompt = input("\nEnter your prompt to analyze: ")

    try:
        print("\nStarting analysis...")
        analysis = await analyzer.analyze(prompt)
        print("\n✨ Final Prompt:")
        print(analysis.final_prompt)

    except TimeoutError as e:
        print(f"\n⏰ Timeout Error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
