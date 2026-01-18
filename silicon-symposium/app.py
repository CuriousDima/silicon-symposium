from litellm import completion

from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.layout import Layout
from rich import print

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

response = completion(
    model="ollama_chat/gemma3:27b",
    messages=[
        {"role": "system", "content": role_nietzsche},
        {"role": "user", "content": first_question},
    ],
    api_base="http://localhost:11434",
    stream=True,
)

full_text = ""
with Live(layout, screen=True, auto_refresh=True, redirect_stderr=False) as live:
    for chunk in response:
        resp_text = chunk["choices"][0]["delta"].content
        if resp_text:
            full_text += chunk["choices"][0]["delta"].content
            # Update the live display with the accumulated text
            layout["conversation"].update(
                Panel(Markdown(full_text), border_style="blue", title="Conversation")
            )
