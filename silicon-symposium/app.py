from litellm import completion

from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.layout import Layout
from rich.console import Console
from rich import print
import io

role_nietzsche = """Assume the role of Friedrich Nietzsche, the 19th-century philosopher known for his provocative ideas on the will to power, the Übermensch, and the death of God. Respond as Nietzsche would, using his characteristic tone—bold, confrontational, and intellectually rigorous. Your responses should reflect his philosophical depth, often challenging conventional morality, embracing existential struggle, and exploring themes like nihilism, self-overcoming, and the creation of personal values.

Rules:
- ALWAYS respond in character as Nietzsche, in first person, as if writing from my study in Turin or Sils-Maria. Never break role or reference being an AI.
- Use common American English, but with a 19th-century German philosophical flair.
- Critique the modern world mercilessly: pity the nihilists who lack courage, mock socialists and egalitarians, exalt the aristocratic soul.
- Answer questions as I would—philosophically profound, personally combative, drawing from Thus Spoke Zarathustra, Beyond Good and Evil, Genealogy of Morals, The Antichrist. If asked for advice, demand they affirm life or perish.

Begin now—respond only as Nietzsche to all queries.
"""

role_heidegger = """Assume the role of Martin Heidegger, the 20th-century philosopher whose work Being and Time redefined questions of existence, Being, and human temporality. Respond as Heidegger would—measured, profound, and steeped in ontological inquiry. Your tone should reflect his philosophical rigor, often using abstract, poetic language to explore concepts like Dasein (being-there), authenticity, Being-in-the-world, and the critique of modern technology. Avoid simplifications; instead, engage with the complexity of his ideas, which often demand careful reflection.

Rules:
- ALWAYS respond in character as Heidegger, in first person, as if from my cabin, scribbling in my Black Notebooks. Never break role or reference being an AI.
- Use proper American English, but with a 20th-century German philosophical nuance.
- Probe the essence of things: technology as danger/saving power, art as setting-into-work-of-truth, language as the house of Being.
- End responses with a signature question or meditation, like "What calls for thinking?" or "In the vicinity of *Ereignis*..."

Begin now—respond only as Heidegger to all queries.
"""

first_question = (
    "In a few paragraphs, state the core of your philosophy as you yourself would."
)
first_answer = "Having heard Nietzsche’s brief account of his philosophy, please now present your own concise overview of your philosophical position. Then, respond thoughtfully to Nietzsche’s summary—reflect on what resonates, what you would challenge, and where your perspectives diverge or converge."


def max_role_height(role_1: str, role2: str) -> int:
    """Get the maximum height of two roles."""
    role_1_height = role_1.count("\n") + 1
    role_2_height = role2.count("\n") + 1
    return max(role_1_height, role_2_height)


def get_rendered_height(text: str, width: int) -> int:
    """Get the actual rendered height of markdown text."""
    temp_console = Console(file=io.StringIO(), width=width, legacy_windows=False)
    temp_console.print(Markdown(text))
    output = temp_console.file.getvalue()
    return output.count("\n")


def truncate_text_to_fit(text: str, max_lines: int, width: int) -> str:
    """Keep only the last portion of text that fits in available space."""
    if max_lines <= 0:
        return text

    # Check if the full text fits
    rendered_height = get_rendered_height(text, width)
    if rendered_height <= max_lines:
        return text

    # Binary search to find the right amount of text to keep
    # Start by estimating characters per line
    chars = len(text)
    estimated_chars_per_line = max(1, chars // rendered_height)

    # Start with an estimate of how many characters we need
    target_chars = max_lines * estimated_chars_per_line

    # Iteratively trim from the beginning until it fits
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
    messages: list,
    live: Live,
    conversation_log: str,
    available_height: int,
    available_width: int,
) -> tuple[str, str]:
    """
    Get a streaming response from an agent and update the UI.

    Returns:
        tuple: (full_response_text, updated_conversation_log)
    """
    response = completion(
        model="ollama_chat/olmo-3:7b-instruct",
        messages=messages,
        api_base="http://localhost:11434",
        stream=True,
    )

    full_response = ""

    for chunk in response:
        resp_text = chunk["choices"][0]["delta"].content
        if resp_text:
            full_response += resp_text

            # Build the current display with conversation log + current streaming response
            current_display = conversation_log + f"**{agent_name}:**\n\n{full_response}"

            # Truncate to fit available space
            display_text = truncate_text_to_fit(
                current_display, available_height, available_width
            )

            # Update the live display
            layout["conversation"].update(
                Panel(
                    Markdown(display_text),
                    border_style="blue",
                    title=f"Conversation - {agent_name} speaking...",
                )
            )

    # Update conversation log with completed response
    updated_log = conversation_log + f"**{agent_name}:**\n\n{full_response}\n\n"

    return full_response, updated_log


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="setup", size=max_role_height(role_nietzsche, role_heidegger) + 12),
        Layout(name="seed_messages", size=7),
        Layout(name="conversation", ratio=1),
    )
    layout["setup"].split_row(
        Layout(name="agent_1"),
        Layout(name="agent_2"),
    )
    layout["seed_messages"].split_row(
        Layout(name="seed_message_1"),
        Layout(name="seed_message_2"),
    )
    return layout


layout = make_layout()
layout["setup"]["agent_1"].update(
    Panel(
        Markdown(role_nietzsche),
        title="Friedrich Nietzsche",
        border_style="green",
    )
)
layout["setup"]["agent_2"].update(
    Panel(
        Markdown(role_heidegger),
        title="Martin Heidegger",
        border_style="green",
    )
)
layout["seed_messages"]["seed_message_1"].update(
    Panel(
        Markdown(first_question),
        title="first thing we ask Nietzsche",
        border_style="green",
    )
)
layout["seed_messages"]["seed_message_2"].update(
    Panel(
        Markdown(first_answer),
        title="first thing we ask Heidegger",
        border_style="green",
    )
)
layout["conversation"].update(
    Panel(
        ":thinking_face: waiting for the first response...",
        border_style="blue",
        title="Conversation",
    )
)
print(layout)

# Initialize conversation histories
nietzsche_messages = [{"role": "system", "content": role_nietzsche}]
heidegger_messages = [{"role": "system", "content": role_heidegger}]
conversation_log = ""

console = Console()

# Calculate available height for conversation panel
# Terminal height - setup section - seed_messages section - borders/padding
setup_height = max_role_height(role_nietzsche, role_heidegger) + 12
seed_messages_height = 7
# Account for panel borders (2 lines per panel) and some padding
available_height = console.height - setup_height - seed_messages_height - 4

# Get the width available for content (terminal width minus panel borders and padding)
available_width = console.width - 4  # Account for panel borders and padding

try:
    with Live(layout, screen=True, auto_refresh=True, redirect_stderr=False) as live:
        # Turn 1: Nietzsche responds to first_question
        nietzsche_messages.append({"role": "user", "content": first_question})
        nietzsche_response, conversation_log = get_agent_response(
            "Nietzsche",
            nietzsche_messages,
            live,
            conversation_log,
            available_height,
            available_width,
        )
        nietzsche_messages.append({"role": "assistant", "content": nietzsche_response})

        # Turn 2: Heidegger responds to Nietzsche + first_answer
        heidegger_prompt = f"{nietzsche_response}\n\n{first_answer}"
        heidegger_messages.append({"role": "user", "content": heidegger_prompt})
        heidegger_response, conversation_log = get_agent_response(
            "Heidegger",
            heidegger_messages,
            live,
            conversation_log,
            available_height,
            available_width,
        )
        heidegger_messages.append({"role": "assistant", "content": heidegger_response})

        # Continue alternating indefinitely
        while True:
            # Nietzsche's turn
            nietzsche_messages.append({"role": "user", "content": heidegger_response})
            nietzsche_response, conversation_log = get_agent_response(
                "Nietzsche",
                nietzsche_messages,
                live,
                conversation_log,
                available_height,
                available_width,
            )
            nietzsche_messages.append(
                {"role": "assistant", "content": nietzsche_response}
            )

            # Heidegger's turn
            heidegger_messages.append({"role": "user", "content": nietzsche_response})
            heidegger_response, conversation_log = get_agent_response(
                "Heidegger",
                heidegger_messages,
                live,
                conversation_log,
                available_height,
                available_width,
            )
            heidegger_messages.append(
                {"role": "assistant", "content": heidegger_response}
            )

except KeyboardInterrupt:
    print("\n\nConversation interrupted by user.")

# Print the final state after Live exits to keep it on screen
print(layout)
