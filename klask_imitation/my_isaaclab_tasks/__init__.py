# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml

# Resolve paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# Try to load extension metadata, or use defaults
try:
    ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
except FileNotFoundError:
    # Default metadata if config file doesn't exist
    ISAACLAB_TASKS_METADATA = {
        "package": {
            "title": "Klask Imitation Tasks",
            "description": "Custom Klask tasks for imitation learning",
            "version": "0.1.0"
        }
    }

# Import your tasks
from .manager_based.klask import *  # noqa: F401, F403

__all__ = ["ISAACLAB_TASKS_METADATA"]
