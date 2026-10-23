"""CodeFix AI — autonomous multi-agent bug resolution."""

import warnings

__version__ = "0.2.0"

# Library notices that do not apply to us: CrewAI checkpointing (we never checkpoint crews; run
# state lives on the Blackboard) and a pydantic-settings forward-reference notice inside a dependency.
warnings.filterwarnings("ignore", message="function callbacks cannot be serialized")
warnings.filterwarnings("ignore", message="Field 'lifespan' has an incomplete definition")
