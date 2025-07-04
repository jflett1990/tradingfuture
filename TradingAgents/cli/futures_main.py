#!/usr/bin/env python3
"""
Futures Trading CLI Interface
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Add the parent directory to the path so we can import from tradingagents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tradingagents.graph.futures_trading_graph import FuturesTradingGraph
from tradingagents.default_config import DEFAULT_CONFIG


class FuturesTradingCLI:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.futures_symbols = self.config["futures_symbols"]
        
    def display_welcome(self):
        """Display welcome message and futures trading info"""
        print("\n" + "="*70)
        print("🚀 FUTURES TRADING AGENTS")
        print("="*70)
        print("Multi-Agent LLM Framework for Futures Trading")
        print("Specialized for Commodity & Financial Futures Markets")
        print("="*70)
        
        print("\n📊 Available Futures Categories:")
        for category, symbols in self.futures_symbols.items():
            print(f"  {category.upper()}: {', '.join(symbols)}")
        
        print("\n⚠️  RISK DISCLAIMER:")
        print("Futures trading involves substantial risk of loss.")
        print("This framework is for research and educational purposes only.")
        print("Not intended as financial advice.")
        print("="*70)
    
    def get_user_inputs(self) -> Dict:
        """Get user inputs for futures trading analysis"""
        print("\n🔧 Configuration Setup")
        print("-" * 30)
        
        # Get futures symbol
        symbol = self.get_futures_symbol()
        
        # Get analysis date
        date = self.get_analysis_date()
        
        # Get analysis depth
        depth = self.get_analysis_depth()
        
        # Get LLM settings
        llm_config = self.get_llm_config()
        
        return {
            "symbol": symbol,
            "date": date,
            "depth": depth,
            "llm_config": llm_config
        }
    
    def get_futures_symbol(self) -> str:
        """Get futures symbol from user input"""
        print("\n📈 Select Futures Contract:")
        print("1. Browse by category")
        print("2. Enter symbol directly")
        
        choice = input("Choice (1-2): ").strip()
        
        if choice == "1":
            return self.browse_symbols_by_category()
        elif choice == "2":
            return self.enter_symbol_directly()
        else:
            print("Invalid choice. Defaulting to browse by category.")
            return self.browse_symbols_by_category()
    
    def browse_symbols_by_category(self) -> str:
        """Browse futures symbols by category"""
        print("\n📊 Select Category:")
        categories = list(self.futures_symbols.keys())
        
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category.upper()}")
        
        try:
            cat_choice = int(input("Category (1-{}): ".format(len(categories)))) - 1
            if 0 <= cat_choice < len(categories):
                selected_category = categories[cat_choice]
                symbols = self.futures_symbols[selected_category]
                
                print(f"\n📋 {selected_category.upper()} Futures:")
                for i, symbol in enumerate(symbols, 1):
                    print(f"{i}. {symbol}")
                
                try:
                    sym_choice = int(input("Symbol (1-{}): ".format(len(symbols)))) - 1
                    if 0 <= sym_choice < len(symbols):
                        return symbols[sym_choice]
                    else:
                        print("Invalid choice. Using first symbol.")
                        return symbols[0]
                except ValueError:
                    print("Invalid input. Using first symbol.")
                    return symbols[0]
            else:
                print("Invalid category. Using energy category.")
                return self.futures_symbols["energy"][0]
        except ValueError:
            print("Invalid input. Using energy category.")
            return self.futures_symbols["energy"][0]
    
    def enter_symbol_directly(self) -> str:
        """Enter futures symbol directly"""
        print("\nEnter futures symbol (e.g., CL, GC, ES):")
        print("Note: =F suffix will be added automatically")
        
        symbol = input("Symbol: ").strip().upper()
        
        if not symbol:
            print("Empty symbol. Using CL (Crude Oil).")
            return "CL"
        
        return symbol
    
    def get_analysis_date(self) -> str:
        """Get analysis date from user"""
        print("\n📅 Analysis Date:")
        print("1. Today")
        print("2. Yesterday") 
        print("3. Custom date")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            return datetime.now().strftime("%Y-%m-%d")
        elif choice == "2":
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif choice == "3":
            date_input = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                # Validate date format
                datetime.strptime(date_input, "%Y-%m-%d")
                return date_input
            except ValueError:
                print("Invalid date format. Using today.")
                return datetime.now().strftime("%Y-%m-%d")
        else:
            print("Invalid choice. Using today.")
            return datetime.now().strftime("%Y-%m-%d")
    
    def get_analysis_depth(self) -> str:
        """Get analysis depth preference"""
        print("\n🔍 Analysis Depth:")
        print("1. Quick (1 debate round)")
        print("2. Standard (2 debate rounds)")
        print("3. Deep (3 debate rounds)")
        
        choice = input("Choice (1-3): ").strip()
        
        depth_map = {
            "1": "quick",
            "2": "standard", 
            "3": "deep"
        }
        
        return depth_map.get(choice, "standard")
    
    def get_llm_config(self) -> Dict:
        """Get LLM configuration"""
        print("\n🤖 LLM Configuration:")
        print("1. Standard (gpt-4o-mini)")
        print("2. Advanced (gpt-4o)")
        print("3. Custom")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            return {
                "deep_think_llm": "gpt-4o-mini",
                "quick_think_llm": "gpt-4o-mini"
            }
        elif choice == "2":
            return {
                "deep_think_llm": "gpt-4o",
                "quick_think_llm": "gpt-4o-mini"
            }
        elif choice == "3":
            deep_model = input("Deep thinking model: ").strip() or "gpt-4o-mini"
            quick_model = input("Quick thinking model: ").strip() or "gpt-4o-mini"
            return {
                "deep_think_llm": deep_model,
                "quick_think_llm": quick_model
            }
        else:
            print("Invalid choice. Using standard configuration.")
            return {
                "deep_think_llm": "gpt-4o-mini",
                "quick_think_llm": "gpt-4o-mini"
            }
    
    def run_analysis(self, inputs: Dict):
        """Run futures trading analysis"""
        print("\n🔄 Starting Futures Trading Analysis...")
        print(f"Symbol: {inputs['symbol']}")
        print(f"Date: {inputs['date']}")
        print(f"Depth: {inputs['depth']}")
        print("-" * 50)
        
        # Update config with user inputs
        config = self.config.copy()
        config.update(inputs["llm_config"])
        
        # Set debate rounds based on depth
        depth_rounds = {
            "quick": 1,
            "standard": 2,
            "deep": 3
        }
        config["max_debate_rounds"] = depth_rounds.get(inputs["depth"], 2)
        
        try:
            # Initialize futures trading graph
            futures_graph = FuturesTradingGraph(debug=True, config=config)
            
            # Run analysis
            state, decision = futures_graph.propagate(inputs["symbol"], inputs["date"])
            
            # Display results
            self.display_results(decision)
            
            # Save results
            self.save_results(decision, inputs)
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            print("Please check your configuration and try again.")
    
    def display_results(self, decision: Dict):
        """Display analysis results"""
        print("\n" + "="*70)
        print("📊 FUTURES TRADING ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\nSymbol: {decision['symbol']}")
        print(f"Date: {decision['date']}")
        print(f"Risk Decision: {decision['risk_decision']}")
        
        print("\n📈 TRADER RECOMMENDATION:")
        print("-" * 40)
        print(decision.get('trader_recommendation', 'N/A'))
        
        print("\n⚠️  RISK ASSESSMENT:")
        print("-" * 40)
        print(decision.get('risk_assessment', 'N/A'))
        
        print("\n" + "="*70)
    
    def save_results(self, decision: Dict, inputs: Dict):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"futures_analysis_{decision['symbol']}_{timestamp}.json"
        
        results = {
            "inputs": inputs,
            "decision": decision,
            "timestamp": timestamp
        }
        
        # Create results directory if it doesn't exist
        results_dir = self.config.get("results_dir", "./results")
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {str(e)}")
    
    def run(self):
        """Main CLI entry point"""
        try:
            self.display_welcome()
            inputs = self.get_user_inputs()
            self.run_analysis(inputs)
        except KeyboardInterrupt:
            print("\n\n👋 Analysis interrupted by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please report this issue if it persists.")


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description="Futures Trading Analysis CLI")
    parser.add_argument("--symbol", help="Futures symbol (e.g., CL, GC, ES)")
    parser.add_argument("--date", help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], 
                        default="standard", help="Analysis depth")
    parser.add_argument("--model", choices=["standard", "advanced"], 
                        default="standard", help="LLM model configuration")
    
    args = parser.parse_args()
    
    cli = FuturesTradingCLI()
    
    if args.symbol and args.date:
        # Non-interactive mode
        llm_config = {
            "deep_think_llm": "gpt-4o" if args.model == "advanced" else "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini"
        }
        
        inputs = {
            "symbol": args.symbol,
            "date": args.date,
            "depth": args.depth,
            "llm_config": llm_config
        }
        
        cli.run_analysis(inputs)
    else:
        # Interactive mode
        cli.run()


if __name__ == "__main__":
    main()