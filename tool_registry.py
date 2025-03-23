import json
import os
from logger import Logger


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        returns: dict,
        raw_json: dict,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns = returns
        self.raw_json = raw_json

    def __str__(self) -> str:
        return f"""
        <tool_definition>
            {self.raw_json}
        </tool_definition>
        """


class ToolRegistry:
    def __init__(self, registry_dir: str = "tool_registry"):
        self.tools = []
        self.registry_dir = registry_dir

    def load_tool(self, tool_path: str):
        with open(tool_path, "r") as f:
            tool_json = json.load(f)
            self.tools.append(
                Tool(**tool_json, raw_json=json.dumps(tool_json, indent=2))
            )

        self.log(f"Loaded tool from {tool_path}")

    def load_all_tools(self):
        for filename in os.listdir(self.registry_dir):
            self.load_tool(os.path.join(self.registry_dir, filename))

        self.log(f"Loaded {len(self.tools)} tools")

    def get_tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def get_tool(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def __str__(self) -> str:
        return (
            "\n".join([str(tool) for tool in self.tools])
            if self.tools
            else "No tools loaded"
        )

    def log(self, message: str):
        Logger.print_system("ToolRegistry", message)
