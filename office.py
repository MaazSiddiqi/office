import sys
from agent_registry import AgentRegistry
from tool_registry import ToolRegistry
from logger import Logger
from ea import ExecutiveAssistant

EXIT_PHRASES = ["exit", "quit", "bye"]
COMMANDS = {
    "exit": "Exit the AI Office",
    "help": "Show this help message",
}


EA_PROMPT_PATH = "prompts/ea.prompt.txt"


class Office:
    def __init__(
        self,
    ):
        self.agent_registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
        self.ea = ExecutiveAssistant(
            EA_PROMPT_PATH,
            {
                "agent_registry": str(self.agent_registry),
                "tool_registry": str(self.tool_registry),
            },
        )

    def load(self):
        """
        Load the agent registry
        """
        self.agent_registry.load()

    def run(self):
        """
        Run the AI Office
        """
        Logger.print_welcome()

        while True:
            running = self.handle_user_input()
            if not running:
                break

        self.stop()

    def handle_user_input(self):
        """
        Handle user input

        Returns:
            bool: True if the main loop should continue, False if it should break
        """
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
        """
        Check if the user input is a command

        Returns:
            bool: True if the input is a command, False otherwise
        """
        return user_input.startswith("/")

    def handle_command(self, command: str, args: list[str]):
        """
        Handle a command

        Args:
            command (str): The command to handle
            args (list[str]): The arguments to the command
        """
        match command:
            case "help":
                Logger.print_system_message(COMMANDS[command])
            case "exit":
                self.stop()
            case _:
                Logger.print_error(f"Unknown command: {command}")

    def handle_user_query(self, user_input: str):
        """
        Handle a user query

        Args:
            user_input (str): The user query to handle
        """
        response = self.ea.ask(user_input)

        for chunk in response:
            Logger.print_llm_stream(chunk)
        Logger.print_llm_stream("\n")

    def stop(self):
        """
        Stop the AI Office
        """
        Logger.print_warning("Shutting down AI Office...")
        sys.exit(0)
