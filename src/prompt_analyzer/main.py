import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv, set_key
from langchain import RunnableSequence
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
                    temperature=0.7
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize models: {str(e)}") from e

    def set_pipeline(self, custom_pipeline: Dict[str, str]):
        """Set custom pipeline if needed"""
        self.pipeline = custom_pipeline

    def analyze(self, prompt: str) -> PromptAnalysis:
        # Analysis
        analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze this prompt's structure and effectiveness:\n\n{prompt}"
        )
        analysis_chain = analysis_prompt | self.models[self.pipeline["analysis"]]

        # Rewrite
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this prompt for clarity:\n\n{prompt}\n\nAnalysis: {initial_analysis}"
        )
        rewrite_chain = rewrite_prompt | self.models[self.pipeline["rewrite"]]

        # Improvements
        improve_prompt = ChatPromptTemplate.from_template(
            "List specific improvements for this prompt:\n\n{rewritten_prompt}"
        )
        improve_chain = improve_prompt | self.models[self.pipeline["improvements"]]

        # Refinement
        refine_prompt = ChatPromptTemplate.from_template(
            "Refine this prompt:\n\n{rewritten_prompt}\n\nSuggested improvements: {potential_improvements}"
        )
        refine_chain = refine_prompt | self.models[self.pipeline["refinement"]]

        # Final
        final_prompt = ChatPromptTemplate.from_template(
            "Create the final optimized version:\n\n{refined_prompt}"
        )
        final_chain = final_prompt | self.models[self.pipeline["final"]]

        # Chain everything together using the pipe operator
        chain = RunnableSequence(
            {
                "initial_analysis": analysis_chain,
                "rewritten_prompt": rewrite_chain,
                "potential_improvements": improve_chain,
                "refined_prompt": refine_chain,
                "final_prompt": final_chain,
            }
        )

        # Run the chain
        results = chain.invoke({"prompt": prompt})

        # Parse improvements into list
        improvements = [
            imp.strip()
            for imp in results["potential_improvements"].content.split("\n")
            if imp.strip() and not imp.strip().isspace()
        ]
        improvements = [imp.lstrip("•-*[] 1234567890.").strip() for imp in improvements]

        return PromptAnalysis(
            initial_analysis=results["initial_analysis"].content,
            rewritten_prompt=results["rewritten_prompt"].content,
            potential_improvements=improvements,
            refined_prompt=results["refined_prompt"].content,
            final_prompt=results["final_prompt"].content,
            models_used={
                stage: self.DEFAULT_MODELS[model_key].name
                for stage, model_key in self.pipeline.items()
            },
        )


def main():
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
        print("\nAnalyzing prompt...")
        analysis = analyzer.analyze(prompt)

        print("\n=== Results ===")
        print("\n📝 Initial Analysis:")
        print(analysis.initial_analysis)

        print("\n✍️ Rewritten Prompt:")
        print(analysis.rewritten_prompt)

        print("\n💡 Potential Improvements:")
        for i, improvement in enumerate(analysis.potential_improvements, 1):
            print(f"{i}. {improvement}")

        print("\n🔄 Refined Prompt:")
        print(analysis.refined_prompt)

        print("\n✨ Final Prompt:")
        print(analysis.final_prompt)

        print("\n🔍 Models Used:")
        for stage, model in analysis.models_used.items():
            print(f"- {stage}: {model}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
