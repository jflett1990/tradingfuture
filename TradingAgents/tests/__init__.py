"""
Testing infrastructure for TradingAgents.

This module provides testing utilities, fixtures, and base classes
for comprehensive testing of the trading system.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment
os.environ["TRADING_ENV"] = "testing"
os.environ["DEBUG"] = "true"