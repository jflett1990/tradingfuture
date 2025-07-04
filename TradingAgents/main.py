"""
Main entry point for TradingAgents.

This example demonstrates using the improved configuration and error handling systems.
"""

import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.utils.exceptions import TradingAgentsError, ConfigurationError
from tradingagents.utils.logging_config import get_logger
from tradingagents.config import get_config_manager

def main():
    """Main execution function with proper error handling."""
    try:
        # Set up environment variables if not already set
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            print("Warning: No API keys found in environment variables.")
            print("Please set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY")
        
        # Option 1: Use new configuration system with environment variables
        # The configuration will be loaded from environment variables automatically
        ta = TradingAgentsGraph(debug=True)
        
        # Option 2: Use custom configuration file (uncomment to use)
        # ta = TradingAgentsGraph(debug=True, config_file="custom_config.json")
        
        # Option 3: Legacy mode with manual config (uncomment to use)
        # from tradingagents.default_config import DEFAULT_CONFIG
        # config = DEFAULT_CONFIG.copy()
        # config["llm_provider"] = "google"
        # config["deep_think_llm"] = "gemini-2.0-flash"
        # config["quick_think_llm"] = "gemini-2.0-flash"
        # ta = TradingAgentsGraph(debug=True, config=config)
        
        # Get logger for this session
        logger = get_logger()
        
        # Run analysis
        print("Starting trading analysis...")
        logger.log_agent_action("main", "starting_analysis", {"ticker": "NVDA", "date": "2024-05-10"})
        
        # Forward propagate
        final_state, decision = ta.propagate("NVDA", "2024-05-10")
        
        print(f"Trading Decision: {decision}")
        logger.log_trading_action("decision", "NVDA", reason=str(decision))
        
        # Optionally reflect and remember (uncomment to use)
        # print("Reflecting on decision...")
        # ta.reflect_and_remember(1000)  # parameter is the position returns
        
        print("Analysis completed successfully!")
        
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        if hasattr(e, 'context'):
            print(f"Context: {e.context}")
        return 1
        
    except TradingAgentsError as e:
        print(f"Trading Agents Error: {e}")
        if hasattr(e, 'context'):
            print(f"Context: {e.context}")
        if hasattr(e, 'error_code'):
            print(f"Error Code: {e.error_code}")
        return 1
        
    except Exception as e:
        print(f"Unexpected Error: {e}")
        logger = get_logger()
        logger.log_error(e, {"phase": "main_execution"})
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
