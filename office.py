import sys
from agent_registry import AgentRegistry
from logger import Logger
from llm import LLM

EXIT_PHRASES = ["exit", "quit", "bye"]
COMMANDS = {
    "exit": "Exit the AI Office",
    "help": "Show this help message",
}


class Office:
    def __init__(self):
        self.agent_registry = AgentRegistry()
        self.llm = LLM(model="llama3.1:latest", max_tokens=2048, temperature=0.7)

    def load(self):
        self.agent_registry.load()

    def run(self):
        Logger.print_welcome()

        while True:
            running = self.handle_user_input()
            if not running:
                break

        self.stop()

    def handle_user_input(self):
        user_input = input(f"{Logger.format_timestamp()} > ").strip()

        if user_input.lower() in EXIT_PHRASES:
            return False

        if self.is_command(user_input):
            command, *args = user_input[1:].split(" ")
            self.handle_command(command, args)
            return True
        else:
            self.handle_user_query(user_input)

        return True

    def is_command(self, user_input: str) -> bool:
        return user_input.startswith("/")

    def handle_command(self, command: str, args: list[str]):
        match command:
            case "help":
                Logger.print_system_message(COMMANDS[command])
            case "exit":
                self.stop()
            case _:
                Logger.print_error(f"Unknown command: {command}")

    def handle_user_query(self, user_input: str):
        response = self.llm.generate_response_stream(user_input)

        for chunk in response:
            Logger.print_llm_stream(chunk)
        Logger.print_llm_stream("\n")

    def stop(self):
        Logger.print_warning("Shutting down AI Office...")
        sys.exit(0)
