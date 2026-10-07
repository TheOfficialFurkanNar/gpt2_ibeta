from dataclasses import dataclass, field
from typing import List


@dataclass
class ChatConfig:
    """Configuration for the CLI chat application."""
    
    model_name: str = "FurkanNar/GPT-2_Instruct-Bigger"
    local_model_path: str = "saved_model"  # If set, uses local model path instead of downloading
    system_prompt: str = "Below is a conversation between a user and a helpful AI assistant."
    max_length: int = 512
    max_new_tokens: int = 100
    temperature: float = 0.7
    num_return_sequences: int = 1
    do_sample: bool = True
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    stop_sequences: List[str] = field(default_factory=lambda: [
        "\nInstruction:",
        "\nResponse:",
        "\nUser:",
        "\nQ:",
        "\nHuman:",
        "### Instruction:",
        "### Response:",
        "###"
    ])


