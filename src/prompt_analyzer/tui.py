from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Input, TextArea, LoadingIndicator, Label
from textual.binding import Binding
from prompt_analyzer.main import PromptAnalyzer
import asyncio


class PromptAnalyzerTUI(App):
    CSS = """
    Screen {
        align: center middle;
    }

    #prompt-container {
        width: 90%;
        height: 90%;
        border: solid green;
        padding: 1;
    }

    #input-prompt {
        dock: top;
        width: 100%;
        margin: 1;
    }

    #result-area {
        width: 100%;
        height: 1fr;
        border: solid $primary-lighten-2;
        margin: 1;
    }

    #status-container {
        height: auto;
        width: 100%;
        align: center middle;
    }

    #stage-label {
        text-align: center;
        padding: 1;
        color: $accent;
    }

    LoadingIndicator {
        width: 100%;
        height: 1;
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "analyze", "Analyze"),
    ]

    def __init__(self):
        super().__init__()
        self.analyzer = PromptAnalyzer()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="prompt-container"):
            yield Input(placeholder="Enter your prompt...", id="input-prompt")
            with Vertical(id="status-container"):
                yield Label("Ready", id="stage-label")
                yield LoadingIndicator(id="loading")
            yield TextArea(id="result-area")

    def on_mount(self) -> None:
        self.query_one(LoadingIndicator).display = False

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        await self.analyze_prompt()

    def action_analyze(self) -> None:
        self.run_worker(self.analyze_prompt())

    def update_stage(self, stage: str) -> None:
        stage_label = self.query_one("#stage-label", Label)
        stage_label.update(f"Stage: {stage}")

    def update_result(self, text: str) -> None:
        result_area = self.query_one("#result-area", TextArea)
        current_text = result_area.text
        result_area.text = current_text + "\n" + text if current_text else text

    async def analyze_prompt(self) -> None:
        prompt = self.query_one("#input-prompt", Input).value
        if not prompt:
            return

        loading = self.query_one(LoadingIndicator)
        result_area = self.query_one("#result-area", TextArea)

        loading.display = True
        result_area.loading = True
        result_area.text = ""  # Clear previous results

        try:
            # Analysis stage
            self.update_stage("Analysis")
            self.update_result("Analyzing prompt...")
            await asyncio.sleep(1)  # Simulate analysis time

            # Rewriting stage
            self.update_stage("Rewriting")
            self.update_result("Rewriting prompt for clarity...")
            await asyncio.sleep(1)

            # Improvements stage
            self.update_stage("Improvements")
            self.update_result("Applying improvements...")
            await asyncio.sleep(1)

            # Refinement stage
            self.update_stage("Refinement")
            self.update_result("Refining final output...")
            await asyncio.sleep(1)

            # Final stage
            self.update_stage("Final")
            analysis = await self.analyzer.analyze(prompt)
            self.update_result("\nFinal Prompt:\n" + analysis.final_prompt)
            self.update_stage("Complete")

        except Exception as e:
            result_area.text = f"Error: {str(e)}"
            self.update_stage("Error")
        finally:
            loading.display = False
            result_area.loading = False


async def main() -> None:
    app = PromptAnalyzerTUI()
    await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
