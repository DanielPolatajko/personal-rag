# src/utils/config.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "rag": {"chunk_size": 1000, "chunk_overlap": 200, "top_k_results": 5},
        "llm": {
            "model_name": "claude-3-haiku-20240307",
            "temperature": 0.1,
            "max_tokens": 1000,
        },
        "embeddings": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        "vector_store": {
            "persist_directory": "./data/vectorstore",
            "collection_name": "research_documents",
        },
        "scraping": {
            "timeout": 30,
            "user_agent": "RAG Research Bot 1.0",
            "max_content_length": 50000,
        },
        "app": {"title": "Personal RAG Research Assistant", "port": 8501},
    }

    # Try to load from YAML file
    if config_path is None:
        # Look for config file in common locations
        possible_paths = [
            "./config/settings.yaml",
            "./settings.yaml",
            os.path.expanduser("~/.rag_app/settings.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                # Merge with default config
                config = deep_merge(default_config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config file {config_path}: {str(e)}")
            config = default_config
    else:
        logger.info("No config file found, using defaults")
        config = default_config

    # Override with environment variables
    config = apply_env_overrides(config)

    # Ensure data directories exist
    create_data_directories(config)

    return config


def deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables should be named like: RAG_SECTION_KEY
    For example: RAG_LLM_MODEL_NAME=claude-3-opus-20240229
    """
    env_mappings = {
        # LLM settings
        "RAG_LLM_MODEL_NAME": ["llm", "model_name"],
        "RAG_LLM_TEMPERATURE": ["llm", "temperature"],
        "RAG_LLM_MAX_TOKENS": ["llm", "max_tokens"],
        # RAG settings
        "RAG_CHUNK_SIZE": ["rag", "chunk_size"],
        "RAG_CHUNK_OVERLAP": ["rag", "chunk_overlap"],
        "RAG_TOP_K_RESULTS": ["rag", "top_k_results"],
        # Vector store settings
        "RAG_PERSIST_DIRECTORY": ["vector_store", "persist_directory"],
        "RAG_COLLECTION_NAME": ["vector_store", "collection_name"],
        # Embedding settings
        "RAG_EMBEDDING_MODEL": ["embeddings", "model_name"],
        # Scraping settings
        "RAG_SCRAPING_TIMEOUT": ["scraping", "timeout"],
        "RAG_SCRAPING_USER_AGENT": ["scraping", "user_agent"],
        # App settings
        "RAG_APP_PORT": ["app", "port"],
    }

    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Navigate to the nested config location
            current = config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Convert value to appropriate type
            final_key = config_path[-1]
            if final_key in ["temperature"]:
                current[final_key] = float(env_value)
            elif final_key in [
                "chunk_size",
                "chunk_overlap",
                "top_k_results",
                "max_tokens",
                "timeout",
                "port",
            ]:
                current[final_key] = int(env_value)
            else:
                current[final_key] = env_value

            logger.info(f"Applied environment override: {env_var}={env_value}")

    return config


def create_data_directories(config: Dict[str, Any]):
    """Create necessary data directories."""
    directories = [
        config["vector_store"]["persist_directory"],
        "./data/documents",
        "./data/metadata",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def save_config(config: Dict[str, Any], config_path: str = "./config/settings.yaml"):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the config file
    """
    try:
        # Ensure config directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")


def get_api_key() -> str:
    """
    Get Anthropic API key from environment or prompt user.

    Returns:
        API key string

    Raises:
        ValueError: If no API key is found
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        # Try alternative environment variable names
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_TOKEN")

    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable."
        )

    return api_key


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required sections
        required_sections = ["rag", "llm", "embeddings", "vector_store"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False

        # Validate ranges
        if not (100 <= config["rag"]["chunk_size"] <= 10000):
            logger.error("chunk_size must be between 100 and 10000")
            return False

        if not (0 <= config["rag"]["chunk_overlap"] < config["rag"]["chunk_size"]):
            logger.error("chunk_overlap must be >= 0 and < chunk_size")
            return False

        if not (0.0 <= config["llm"]["temperature"] <= 2.0):
            logger.error("temperature must be between 0.0 and 2.0")
            return False

        if not (1 <= config["llm"]["max_tokens"] <= 8000):
            logger.error("max_tokens must be between 1 and 8000")
            return False

        # Check if directories are writable
        persist_dir = Path(config["vector_store"]["persist_directory"])
        if persist_dir.exists() and not os.access(persist_dir, os.W_OK):
            logger.error(f"Vector store directory not writable: {persist_dir}")
            return False

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation error: {str(e)}")
        return False


# Configuration templates for different use cases
CONFIG_TEMPLATES = {
    "development": {
        "rag": {"chunk_size": 500, "chunk_overlap": 100, "top_k_results": 3},
        "llm": {"temperature": 0.2, "max_tokens": 500},
        "vector_store": {"persist_directory": "./dev_data/vectorstore"},
    },
    "production": {
        "rag": {"chunk_size": 1200, "chunk_overlap": 200, "top_k_results": 7},
        "llm": {"temperature": 0.05, "max_tokens": 1500},
        "vector_store": {"persist_directory": "/data/vectorstore"},
    },
    "research": {
        "rag": {"chunk_size": 1500, "chunk_overlap": 300, "top_k_results": 10},
        "llm": {"temperature": 0.1, "max_tokens": 2000},
        "embeddings": {"model_name": "sentence-transformers/all-mpnet-base-v2"},
    },
}


def create_config_from_template(
    template_name: str, output_path: str = "./config/settings.yaml"
):
    """
    Create a configuration file from a template.

    Args:
        template_name: Name of the template to use
        output_path: Where to save the configuration file
    """
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(CONFIG_TEMPLATES.keys())}"
        )

    # Load default config and merge with template
    base_config = load_config(config_path=None)  # Get defaults
    template_config = CONFIG_TEMPLATES[template_name]

    merged_config = deep_merge(base_config, template_config)
    save_config(merged_config, output_path)

    logger.info(
        f"Created configuration from template '{template_name}' at {output_path}"
    )


if __name__ == "__main__":
    # CLI for config management
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "validate":
            config = load_config()
            is_valid = validate_config(config)
            print(f"Configuration is {'valid' if is_valid else 'invalid'}")
            sys.exit(0 if is_valid else 1)

        elif command == "create-template" and len(sys.argv) > 2:
            template_name = sys.argv[2]
            output_path = sys.argv[3] if len(sys.argv) > 3 else "./config/settings.yaml"
            create_config_from_template(template_name, output_path)

        else:
            print(
                "Usage: python config.py [validate|create-template TEMPLATE_NAME [OUTPUT_PATH]]"
            )
            sys.exit(1)
    else:
        # Just load and print config
        config = load_config()
        print(yaml.dump(config, default_flow_style=False, indent=2))
