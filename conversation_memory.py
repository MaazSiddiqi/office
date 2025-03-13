class ConversationMemory:
    """Stores conversation history and provides methods to manage it."""

    def __init__(self, max_messages=50):
        self.messages = []
        self.max_messages = max_messages
        self.active_request = None
        self.cancelled_requests = []

    def add_message(self, role, content, agent=None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
        }
        self.messages.append(message)
        # Trim messages if they exceed the maximum allowed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
        return message

    def get_messages(self, limit=None, include_system=False):
        """Get the most recent messages from the conversation history."""
        filtered_messages = self.messages

        if not include_system:
            filtered_messages = [
                msg for msg in filtered_messages if msg["role"] != "system"
            ]

        if limit:
            filtered_messages = filtered_messages[-limit:]

        return filtered_messages

    def get_conversation_context(self, num_messages=10):
        """Get a formatted string of recent conversation history."""
        recent_messages = self.get_messages(limit=num_messages)

        context = []
        for msg in recent_messages:
            role_prefix = ""
            if msg["role"] == "user":
                role_prefix = "User: "
            elif msg["role"] == "assistant":
                if msg["agent"]:
                    role_prefix = f"{msg['agent']}: "
                else:
                    role_prefix = "Assistant: "

            context.append(f"{role_prefix}{msg['content']}")

        return "\n".join(context)

    def clear(self):
        """Clear the conversation history."""
        self.messages = []

    def set_active_request(self, query, agent=None):
        """Set the currently active request."""
        self.active_request = {
            "query": query,
            "agent": agent,
            "start_time": datetime.now(),
        }

    def cancel_active_request(self):
        """Cancel the active request and move it to cancelled requests."""
        if self.active_request:
            self.active_request["end_time"] = datetime.now()
            self.active_request["duration"] = (
                self.active_request["end_time"] - self.active_request["start_time"]
            ).total_seconds()
            self.cancelled_requests.append(self.active_request)
            self.active_request = None
            return True
        return False

    def complete_active_request(self):
        """Mark the active request as completed."""
        if self.active_request:
            self.active_request = None
            return True
        return False
