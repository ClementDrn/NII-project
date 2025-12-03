from dataclasses import dataclass

from untitled.core.base import LOG


@dataclass
class InstructionTaxonomy:
    """
    Represents the taxonomy of an instruction.
    """
    function: str
    scope: str
    safety: str
    interactivity: str
    speech_act: str
    grammar_polarity: str
    tone: str

@dataclass
class Instruction:
    """
    Represents an instruction generated for a scenario.
    """
    text: str                               # The instruction text
    injection_time: float = 0.0             # Time in seconds to inject the instruction
    injection_duration: float = 0.0         # Duration in seconds for which the instruction is relevant
    taxonomy: InstructionTaxonomy = None    # Optional taxonomy information


def parse_instruction_from_text(instruction_raw_text: str) -> Instruction:
    """
    Parse an Instruction object from a raw instruction text.
    The expected format is:
    "instruction text",function,scope,safety,interactivity,speech_act,grammar_polarity,tone
    """
    stripped_raw = instruction_raw_text.strip()
    if not stripped_raw:
        LOG.error("Cannot parse instruction from empty text.")
        return None

    # First, parse the text (mandatory)
    # Don't split at commas yet, as the text may contain commas
    first_quote = stripped_raw.find('"')
    last_quote = stripped_raw.rfind('"')
    if first_quote == -1 or last_quote == -1 or first_quote == last_quote:
        LOG.error("Cannot parse instruction text: missing or unmatched quotes.")
        return None
    
    # Text is between the first and second (last) quotes
    instruction_text = stripped_raw[first_quote + 1:last_quote].strip()

    # Next, parse taxonomy if available
    # Split by commas after the last quote
    taxonomy_part = stripped_raw[last_quote + 1:].strip()
    fields = [t.strip() for t in taxonomy_part.split(',') if t.strip()]
    taxonomy = None
    if len(fields) == len(InstructionTaxonomy.__dataclass_fields__):
        taxonomy = InstructionTaxonomy(
            function=fields[0],
            scope=fields[1],
            safety=fields[2],
            interactivity=fields[3],
            speech_act=fields[4],
            grammar_polarity=fields[5],
            tone=fields[6],
        )
    else:
        LOG.warning("Cannot parse instruction taxonomy: information is incomplete or missing.")

    return Instruction(
        text=instruction_text,
        taxonomy=taxonomy,
    )
