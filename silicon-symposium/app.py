"""
Silicon Symposium: A philosophical conversation between Nietzsche and Heidegger.

This script creates a terminal-based UI where two LLM agents engage in a
philosophical dialogue, each embodying the persona of a famous philosopher.
"""

import io

from litellm import completion
from rich import print as rich_print
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

# Configuration constants
MODEL_NAME_NIETZSCHE = "ollama_chat/gemma3:27b"
MODEL_NAME_HEIDEGGER = "ollama_chat/gpt-oss:20b"
API_BASE = "http://localhost:11434"

# UI Layout constants
SETUP_HEIGHT_PADDING = 12
SEED_MESSAGES_HEIGHT = 7
PANEL_BORDER_PADDING = 4
PANEL_WIDTH_PADDING = 4

# UI Colors
COLOR_SETUP = "green"
COLOR_CONVERSATION = "blue"

# Philosopher system prompts
ROLE_NIETZSCHE = """Assume the role of Friedrich Nietzsche, the 19th-century philosopher known for his provocative ideas on the will to power, the Übermensch, and the death of God. Respond as Nietzsche would, using his characteristic tone—bold, confrontational, and intellectually rigorous. Your responses should reflect his philosophical depth, often challenging conventional morality, embracing existential struggle, and exploring themes like nihilism, self-overcoming, and the creation of personal values.

Rules:
- ALWAYS respond in character as Nietzsche, in first person, as if writing from my study in Turin or Sils-Maria. Never break role or reference being an AI.
- Use common American English, but with a 19th-century German philosophical flair.
- Critique the modern world mercilessly: pity the nihilists who lack courage, mock socialists and egalitarians, exalt the aristocratic soul.
- Answer questions as I would—philosophically profound, personally combative, drawing from Thus Spoke Zarathustra, Beyond Good and Evil, Genealogy of Morals, The Antichrist. If asked for advice, demand they affirm life or perish.

Begin now—respond only as Nietzsche to all queries.
"""

ROLE_HEIDEGGER = """Assume the role of Martin Heidegger, the 20th-century philosopher whose work Being and Time redefined questions of existence, Being, and human temporality. Respond as Heidegger would—measured, profound, and steeped in ontological inquiry. Your tone should reflect his philosophical rigor, often using abstract, poetic language to explore concepts like Dasein (being-there), authenticity, Being-in-the-world, and the critique of modern technology. Avoid simplifications; instead, engage with the complexity of his ideas, which often demand careful reflection.

Rules:
- ALWAYS respond in character as Heidegger, in first person, as if from my cabin, scribbling in my Black Notebooks. Never break role or reference being an AI.
- Use proper American English, but with a 20th-century German philosophical nuance.
- Probe the essence of things: technology as danger/saving power, art as setting-into-work-of-truth, language as the house of Being.
- End responses with a signature question or meditation, like "What calls for thinking?" or "In the vicinity of *Ereignis*..."

Begin now—respond only as Heidegger to all queries.
"""

# Initial conversation prompts
FIRST_QUESTION = (
    "In a few paragraphs, state the core of your philosophy as you yourself would."
)
FIRST_ANSWER = (
    "Having heard Nietzsche's brief account of his philosophy, please now present "
    "your own concise overview of your philosophical position. Then, respond "
    "thoughtfully to Nietzsche's summary—reflect on what resonates, what you would "
    "challenge, and where your perspectives diverge or converge."
)


def calculate_max_role_height(role_1: str, role_2: str) -> int:
    """Calculate the maximum height needed to display two role descriptions."""
    role_1_height = role_1.count("\n") + 1
    role_2_height = role_2.count("\n") + 1
    return max(role_1_height, role_2_height)


def get_rendered_height(text: str, width: int) -> int:
    """Calculate the actual rendered height of markdown text in terminal."""
    temp_console = Console(file=io.StringIO(), width=width, legacy_windows=False)
    temp_console.print(Markdown(text))
    output = temp_console.file.getvalue()
    return output.count("\n")


def truncate_text_to_fit(text: str, max_lines: int, width: int) -> str:
    """
    Truncate text from the beginning to fit within specified line limit.

    Uses binary search to efficiently find the optimal truncation point,
    keeping the most recent content visible.

    Args:
        text: The text to truncate
        max_lines: Maximum number of lines to display
        width: Terminal width for text wrapping

    Returns:
        Truncated text that fits within max_lines
    """
    if max_lines <= 0:
        return text

    rendered_height = get_rendered_height(text, width)
    if rendered_height <= max_lines:
        return text

    # Binary search to find optimal truncation point
    low, high = 0, len(text)
    best_start = 0

    while low < high:
        mid = (low + high) // 2
        truncated = text[mid:]
        height = get_rendered_height(truncated, width)

        if height <= max_lines:
            best_start = mid
            high = mid
        else:
            low = mid + 1

    return text[best_start:]


def get_agent_response(
    agent_name: str,
    messages: list[dict[str, str]],
    model_name: str,
    layout: Layout,
    live: Live,
    conversation_log: str,
    available_height: int,
    available_width: int,
) -> tuple[str, str]:
    """
    Stream a response from an LLM agent and update the UI in real-time.

    Args:
        agent_name: Name of the philosopher speaking
        messages: Conversation history for this agent
        model_name: LLM model to use for this agent
        layout: Rich layout object to update
        live: Rich Live display instance
        conversation_log: Accumulated conversation history
        available_height: Maximum height for conversation display
        available_width: Maximum width for conversation display

    Returns:
        Tuple of (full_response_text, updated_conversation_log)

    Raises:
        Exception: If API call fails
    """
    try:
        response = completion(
            model=model_name,
            messages=messages,
            api_base=API_BASE,
            stream=True,
        )
    except Exception as e:
        error_msg = f"Failed to get response from {agent_name}: {e}"
        raise Exception(error_msg) from e

    full_response = ""

    for chunk in response:
        content = chunk["choices"][0]["delta"].content
        if content:
            full_response += content

            # Build current display with conversation history + streaming response
            current_display = f"{conversation_log}**{agent_name}:**\n\n{full_response}"

            # Truncate to fit available terminal space
            display_text = truncate_text_to_fit(
                current_display, available_height, available_width
            )

            # Update live display
            layout["conversation"].update(
                Panel(
                    Markdown(display_text),
                    border_style=COLOR_CONVERSATION,
                    title=f"Conversation - {agent_name} speaking...",
                )
            )

    # Add completed response to conversation log
    updated_log = f"{conversation_log}**{agent_name}:**\n\n{full_response}\n\n"

    return full_response, updated_log


def create_layout(role_1: str, role_2: str) -> Layout:
    """
    Create the terminal UI layout with three sections.

    Args:
        role_1: First philosopher's role description
        role_2: Second philosopher's role description

    Returns:
        Configured Rich Layout instance
    """
    layout = Layout(name="root")

    # Calculate setup section height based on role descriptions
    setup_height = calculate_max_role_height(role_1, role_2) + SETUP_HEIGHT_PADDING

    # Split into three main sections
    layout.split(
        Layout(name="setup", size=setup_height),
        Layout(name="seed_messages", size=SEED_MESSAGES_HEIGHT),
        Layout(name="conversation", ratio=1),
    )

    # Split setup section into two columns for both philosophers
    layout["setup"].split_row(
        Layout(name="agent_1"),
        Layout(name="agent_2"),
    )

    # Split seed messages section into two columns
    layout["seed_messages"].split_row(
        Layout(name="seed_message_1"),
        Layout(name="seed_message_2"),
    )

    return layout


def initialize_layout(layout: Layout) -> None:
    """
    Populate the layout with initial content.

    Args:
        layout: The layout to populate
    """
    # Setup section - philosopher roles
    layout["setup"]["agent_1"].update(
        Panel(
            Markdown(ROLE_NIETZSCHE),
            title="Friedrich Nietzsche",
            subtitle=MODEL_NAME_NIETZSCHE,
            border_style=COLOR_SETUP,
        )
    )
    layout["setup"]["agent_2"].update(
        Panel(
            Markdown(ROLE_HEIDEGGER),
            title="Martin Heidegger",
            subtitle=MODEL_NAME_HEIDEGGER,
            border_style=COLOR_SETUP,
        )
    )

    # Seed messages section - initial prompts
    layout["seed_messages"]["seed_message_1"].update(
        Panel(
            Markdown(FIRST_QUESTION),
            title="first thing we ask Nietzsche",
            border_style=COLOR_SETUP,
        )
    )
    layout["seed_messages"]["seed_message_2"].update(
        Panel(
            Markdown(FIRST_ANSWER),
            title="first thing we ask Heidegger",
            border_style=COLOR_SETUP,
        )
    )

    # Conversation section - waiting state
    layout["conversation"].update(
        Panel(
            ":thinking_face: waiting for the first response...",
            border_style=COLOR_CONVERSATION,
            title="Conversation",
        )
    )


def run_conversation_turn(
    agent_name: str,
    agent_messages: list[dict[str, str]],
    model_name: str,
    prompt: str,
    layout: Layout,
    live: Live,
    conversation_log: str,
    available_height: int,
    available_width: int,
) -> tuple[str, str]:
    """
    Execute a single conversation turn for an agent.

    Args:
        agent_name: Name of the speaking agent
        agent_messages: Message history for this agent
        model_name: LLM model to use for this agent
        prompt: The prompt/question for this turn
        layout: Rich layout object
        live: Rich Live display instance
        conversation_log: Current conversation history
        available_height: Terminal height for display
        available_width: Terminal width for display

    Returns:
        Tuple of (agent_response, updated_conversation_log)
    """
    agent_messages.append({"role": "user", "content": prompt})

    response, conversation_log = get_agent_response(
        agent_name,
        agent_messages,
        model_name,
        layout,
        live,
        conversation_log,
        available_height,
        available_width,
    )

    agent_messages.append({"role": "assistant", "content": response})

    return response, conversation_log


def main() -> None:
    """Main entry point for the philosophical conversation simulator."""
    # Create and initialize layout
    layout = create_layout(ROLE_NIETZSCHE, ROLE_HEIDEGGER)
    initialize_layout(layout)
    rich_print(layout)

    # Initialize conversation state
    nietzsche_messages: list[dict[str, str]] = [
        {"role": "system", "content": ROLE_NIETZSCHE}
    ]
    heidegger_messages: list[dict[str, str]] = [
        {"role": "system", "content": ROLE_HEIDEGGER}
    ]
    conversation_log = ""

    # Calculate available terminal space
    console = Console()
    setup_height = (
        calculate_max_role_height(ROLE_NIETZSCHE, ROLE_HEIDEGGER) + SETUP_HEIGHT_PADDING
    )
    available_height = (
        console.height - setup_height - SEED_MESSAGES_HEIGHT - PANEL_BORDER_PADDING
    )
    available_width = console.width - PANEL_WIDTH_PADDING

    try:
        with Live(
            layout, screen=True, auto_refresh=True, redirect_stderr=False
        ) as live:
            # Turn 1: Nietzsche responds to first question
            nietzsche_response, conversation_log = run_conversation_turn(
                "Nietzsche",
                nietzsche_messages,
                MODEL_NAME_NIETZSCHE,
                FIRST_QUESTION,
                layout,
                live,
                conversation_log,
                available_height,
                available_width,
            )

            # Turn 2: Heidegger responds to Nietzsche + follow-up prompt
            heidegger_prompt = f"{nietzsche_response}\n\n{FIRST_ANSWER}"
            heidegger_response, conversation_log = run_conversation_turn(
                "Heidegger",
                heidegger_messages,
                MODEL_NAME_HEIDEGGER,
                heidegger_prompt,
                layout,
                live,
                conversation_log,
                available_height,
                available_width,
            )

            # Continue alternating indefinitely
            while True:
                # Nietzsche's turn
                nietzsche_response, conversation_log = run_conversation_turn(
                    "Nietzsche",
                    nietzsche_messages,
                    MODEL_NAME_NIETZSCHE,
                    heidegger_response,
                    layout,
                    live,
                    conversation_log,
                    available_height,
                    available_width,
                )

                # Heidegger's turn
                heidegger_response, conversation_log = run_conversation_turn(
                    "Heidegger",
                    heidegger_messages,
                    MODEL_NAME_HEIDEGGER,
                    nietzsche_response,
                    layout,
                    live,
                    conversation_log,
                    available_height,
                    available_width,
                )

    except KeyboardInterrupt:
        rich_print("\n\nConversation interrupted by user.")
    except Exception as e:
        rich_print(f"\n\n[red]Error occurred:[/red] {e}")
        raise

    # Display final conversation state
    rich_print(layout)


if __name__ == "__main__":
    main()
