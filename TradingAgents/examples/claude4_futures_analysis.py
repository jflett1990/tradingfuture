"""Claude 4 Futures Analysis Example

This script demonstrates how to use Claude 4 (claude-3-5-sonnet-20241022) 
for advanced futures trading analysis with the IB integration.

Usage:
    python examples/claude4_futures_analysis.py --symbol CL --live
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.graph.ib_enhanced_futures_graph import create_ib_enhanced_graph
from tradingagents.default_config import DEFAULT_CONFIG
import structlog

# Setup logging
log = structlog.get_logger("claude4_example")

async def run_claude4_analysis(symbol: str = "CL", live_trading: bool = False):
    """Run futures analysis using Claude 4."""
    
    # Verify Claude 4 configuration
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your_key_here'")
        return
    
    # Show configuration
    config = DEFAULT_CONFIG.copy()
    print("\n🤖 Claude 4 Futures Trading Analysis")
    print("="*50)
    print(f"Model: {config['deep_think_llm']}")
    print(f"Symbol: {symbol}")
    print(f"Mode: {'LIVE TRADING' if live_trading else 'PAPER TRADING'}")
    print(f"API Backend: {config['backend_url']}")
    print("="*50)
    
    try:
        # Create the enhanced graph with Claude 4
        graph = await create_ib_enhanced_graph(
            paper_trading=not live_trading,
            config=config,
            debug=True
        )
        
        print(f"\n📊 Running analysis for {symbol}...")
        
        # Execute analysis
        decision = await graph.propagate(symbol, "2025-01-15")
        
        # Display results
        print("\n🎯 CLAUDE 4 ANALYSIS RESULTS")
        print("="*50)
        print(f"Symbol: {decision['symbol']}")
        print(f"Action: {decision['action']}")
        print(f"Confidence: {decision['confidence']:.1%}")
        print(f"Strategy: {decision['strategy_used']}")
        print(f"Live Price: ${decision['live_price'] or 'N/A'}")
        print(f"Position Size: {decision['position_size']:.3f} contracts")
        print(f"Execution Status: {decision['execution_status']}")
        print(f"Daily P&L: ${decision['daily_pnl']:,.2f}")
        print("\n💭 Claude 4 Reasoning:")
        print("-" * 30)
        print(decision['reasoning'])
        print("\n🛡️ Risk Assessment:")
        print("-" * 30)
        print(decision['risk_assessment'])
        
        if decision.get('order_id'):
            print(f"\n📋 Order ID: {decision['order_id']}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as exc:
        print(f"\n❌ Analysis failed: {exc}")
        log.error("claude4_analysis_failed", symbol=symbol, error=str(exc))

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude 4 Futures Analysis")
    parser.add_argument("--symbol", default="CL", help="Futures symbol (default: CL)")
    parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    
    args = parser.parse_args()
    
    # Run analysis
    asyncio.run(run_claude4_analysis(args.symbol, args.live))

if __name__ == "__main__":
    main()