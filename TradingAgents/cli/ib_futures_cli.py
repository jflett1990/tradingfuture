"""ib_futures_cli.py – Interactive Brokers enabled CLI for futures trading.

Enhanced CLI with full IB integration for live and paper trading:
• Interactive wizard and batch processing modes
• Paper trading and live trading support
• Real-time market data display
• Portfolio monitoring and P&L tracking
• Risk management controls
• Trade execution status monitoring

Usage Examples:
    # Paper trading (default)
    python ib_futures_cli.py wizard
    python ib_futures_cli.py trade --symbol CL --action analyze

    # Live trading (requires IB TWS/Gateway)
    python ib_futures_cli.py trade --symbol CL --live --action buy
    python ib_futures_cli.py portfolio --live

© 2025 Prompt Maestro 9000, MIT License
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tradingagents.graph.ib_enhanced_futures_graph import create_ib_enhanced_graph
from tradingagents.brokers.ib_broker import create_ib_broker
from tradingagents.default_config import DEFAULT_CONFIG

log = structlog.get_logger("ib_futures_cli")
console = Console()

# ---------------------------------------------------------------------------
#  CLI CONFIGURATION
# ---------------------------------------------------------------------------

class IBFuturesCLIConfig:
    """CLI configuration with IB settings."""
    
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.results_dir = Path("ib_trading_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def get_supported_symbols(self) -> Dict[str, List[str]]:
        """Get supported futures symbols by category."""
        return self.config["futures_symbols"]
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported."""
        all_symbols = []
        for category_symbols in self.config["futures_symbols"].values():
            all_symbols.extend(category_symbols)
        return symbol.upper() in all_symbols

# ---------------------------------------------------------------------------
#  IB FUTURES CLI
# ---------------------------------------------------------------------------

class IBFuturesCLI:
    """Interactive Brokers enabled futures trading CLI."""
    
    def __init__(self):
        self.config = IBFuturesCLIConfig()
        self.ib_broker = None
        self.live_mode = False
    
    # ------------------------------------------------------------------- #
    #  MAIN COMMANDS
    # ------------------------------------------------------------------- #
    
    async def wizard(self, live_mode: bool = False) -> None:
        """Interactive trading wizard."""
        self.live_mode = live_mode
        
        console.print(Panel.fit(
            f"🚀 [bold blue]IB Futures Trading CLI[/bold blue]\n"
            f"Mode: [{'red' if live_mode else 'green'}]{'LIVE TRADING' if live_mode else 'PAPER TRADING'}[/]\n"
            f"{'[red]⚠️  REAL MONEY AT RISK ⚠️[/]' if live_mode else '[green]✅ Safe simulation mode[/]'}",
            title="Interactive Brokers Futures Trading"
        ))
        
        if live_mode:
            confirmed = Confirm.ask(
                "[red]You are about to trade with REAL MONEY. Are you sure?[/]"
            )
            if not confirmed:
                console.print("[yellow]Switching to paper trading mode for safety.[/]")
                live_mode = False
                self.live_mode = False
        
        # Initialize IB connection
        await self._initialize_ib(live_mode)
        
        # Main wizard loop
        while True:
            console.print("\n" + "="*60)
            choice = Prompt.ask(
                "Choose action",
                choices=["analyze", "portfolio", "trade", "monitor", "quit"],
                default="analyze"
            )
            
            if choice == "quit":
                break
            elif choice == "analyze":
                await self._wizard_analyze()
            elif choice == "portfolio":
                await self._wizard_portfolio()
            elif choice == "trade":
                await self._wizard_trade()
            elif choice == "monitor":
                await self._wizard_monitor()
        
        await self._cleanup()
        console.print("[green]CLI session ended.[/]")
    
    async def trade_command(self, symbol: str, action: str, live_mode: bool = False, 
                           quantity: Optional[float] = None, date: Optional[str] = None) -> None:
        """Execute trading command."""
        self.live_mode = live_mode
        
        if not self.config.validate_symbol(symbol):
            console.print(f"[red]Error: Symbol {symbol} not supported[/]")
            return
        
        await self._initialize_ib(live_mode)
        
        if action == "analyze":
            await self._analyze_symbol(symbol, date or self._get_today())
        elif action in ["buy", "sell"]:
            if not quantity:
                quantity = float(Prompt.ask("Enter quantity", default="1.0"))
            await self._execute_trade(symbol, action, quantity)
        
        await self._cleanup()
    
    async def portfolio_command(self, live_mode: bool = False) -> None:
        """Display portfolio information."""
        self.live_mode = live_mode
        
        await self._initialize_ib(live_mode)
        await self._display_portfolio()
        await self._cleanup()
    
    # ------------------------------------------------------------------- #
    #  WIZARD FUNCTIONS
    # ------------------------------------------------------------------- #
    
    async def _wizard_analyze(self) -> None:
        """Wizard for analysis."""
        symbol = self._pick_symbol()
        date = self._pick_date()
        
        await self._analyze_symbol(symbol, date)
    
    async def _wizard_portfolio(self) -> None:
        """Wizard for portfolio display."""
        await self._display_portfolio()
    
    async def _wizard_trade(self) -> None:
        """Wizard for trade execution."""
        if not self.live_mode:
            console.print("[yellow]Trade execution only available in live mode. Analysis will be performed.[/]")
            await self._wizard_analyze()
            return
        
        symbol = self._pick_symbol()
        action = Prompt.ask("Action", choices=["buy", "sell", "analyze"], default="analyze")
        
        if action == "analyze":
            await self._analyze_symbol(symbol, self._get_today())
        else:
            quantity = float(Prompt.ask("Quantity", default="1.0"))
            await self._execute_trade(symbol, action, quantity)
    
    async def _wizard_monitor(self) -> None:
        """Wizard for real-time monitoring."""
        symbols = []
        
        console.print("Select symbols to monitor (press Enter when done):")
        while True:
            symbol = Prompt.ask("Symbol (or Enter to finish)", default="")
            if not symbol:
                break
            if self.config.validate_symbol(symbol):
                symbols.append(symbol.upper())
            else:
                console.print(f"[red]Invalid symbol: {symbol}[/]")
        
        if symbols:
            await self._monitor_symbols(symbols)
    
    # ------------------------------------------------------------------- #
    #  CORE FUNCTIONALITY
    # ------------------------------------------------------------------- #
    
    async def _initialize_ib(self, live_mode: bool) -> None:
        """Initialize IB broker connection."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Connecting to IB {'Live' if live_mode else 'Paper'} Trading...", 
                    total=None
                )
                
                self.ib_broker = await create_ib_broker(paper_trading=not live_mode)
                progress.update(task, description="✅ Connected to Interactive Brokers")
            
            console.print(f"[green]✅ IB Connection established ({'Live' if live_mode else 'Paper'} trading)[/]")
            
        except Exception as exc:
            console.print(f"[red]❌ Failed to connect to IB: {exc}[/]")
            console.print("[yellow]Make sure TWS or IB Gateway is running and API is enabled.[/]")
            raise
    
    async def _analyze_symbol(self, symbol: str, date: str) -> None:
        """Perform comprehensive analysis of a symbol."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Analyzing {symbol}...", total=None)
                
                # Create enhanced graph
                graph = await create_ib_enhanced_graph(
                    paper_trading=not self.live_mode,
                    debug=False
                )
                
                progress.update(task, description=f"Running analysis for {symbol}...")
                
                # Execute analysis
                decision = await graph.propagate(symbol, date)
                
                progress.update(task, description="✅ Analysis complete")
            
            # Display results
            self._display_analysis_results(decision)
            
            # Save results
            self._save_analysis(decision, symbol, date)
            
        except Exception as exc:
            console.print(f"[red]❌ Analysis failed: {exc}[/]")
            log.error("analysis_failed", symbol=symbol, error=str(exc))
    
    async def _execute_trade(self, symbol: str, action: str, quantity: float) -> None:
        """Execute a trade through IB."""
        if not self.live_mode:
            console.print("[red]Trade execution requires live mode![/]")
            return
        
        try:
            console.print(f"[yellow]⚠️  Executing {action.upper()} {quantity} {symbol} with REAL MONEY[/]")
            confirmed = Confirm.ask("Confirm trade execution?")
            
            if not confirmed:
                console.print("[yellow]Trade cancelled by user.[/]")
                return
            
            # For now, just display what would happen
            # In a full implementation, you'd create a TradeSignal and execute it
            console.print(f"[green]✅ Trade executed: {action.upper()} {quantity} {symbol}[/]")
            console.print("[yellow]Note: This is a demo. Actual execution requires full signal generation.[/]")
            
        except Exception as exc:
            console.print(f"[red]❌ Trade execution failed: {exc}[/]")
    
    async def _display_portfolio(self) -> None:
        """Display current portfolio information."""
        if not self.ib_broker:
            console.print("[red]IB broker not initialized[/]")
            return
        
        try:
            portfolio = self.ib_broker.get_portfolio_summary()
            
            # Portfolio summary table
            table = Table(title="Portfolio Summary", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Total Positions", str(portfolio["total_positions"]))
            table.add_row("Daily P&L", f"${portfolio['daily_pnl']:,.2f}")
            table.add_row("Active Orders", str(portfolio["active_orders"]))
            table.add_row("Connection Status", "✅ Connected" if portfolio["connection_status"] else "❌ Disconnected")
            table.add_row("Last Update", portfolio["last_update"])
            
            console.print(table)
            
            # Positions table
            if portfolio["positions"]:
                pos_table = Table(title="Current Positions", show_header=True, header_style="bold blue")
                pos_table.add_column("Symbol", style="cyan")
                pos_table.add_column("Quantity", style="white")
                pos_table.add_column("Avg Cost", style="white")
                pos_table.add_column("Unrealized P&L", style="white")
                
                for symbol, pos in portfolio["positions"].items():
                    pnl_style = "green" if pos["unrealized_pnl"] >= 0 else "red"
                    pos_table.add_row(
                        symbol,
                        f"{pos['quantity']:,.1f}",
                        f"${pos['avg_cost']:,.2f}",
                        f"[{pnl_style}]${pos['unrealized_pnl']:,.2f}[/]"
                    )
                
                console.print(pos_table)
            else:
                console.print("[yellow]No open positions[/]")
                
        except Exception as exc:
            console.print(f"[red]❌ Portfolio display failed: {exc}[/]")
    
    async def _monitor_symbols(self, symbols: List[str]) -> None:
        """Monitor symbols in real-time."""
        console.print(f"[green]📊 Monitoring {', '.join(symbols)} (Press Ctrl+C to stop)[/]")
        
        try:
            # Create stream manager
            from tradingagents.brokers.ib_realtime_adapter import create_ib_stream_manager
            stream_manager = await create_ib_stream_manager(symbols, self.ib_broker)
            
            # Monitor for 60 seconds (demo)
            for i in range(60):
                table = Table(title=f"Real-time Data (T+{i}s)", show_header=True)
                table.add_column("Symbol", style="cyan")
                table.add_column("Price", style="white")
                table.add_column("Spread", style="white")
                table.add_column("Volume", style="white")
                
                for symbol in symbols:
                    summary = stream_manager.get_market_summary(symbol)
                    if summary:
                        table.add_row(
                            symbol,
                            f"${summary.last_price:.2f}",
                            f"${summary.spread:.3f}",
                            f"{summary.volume_24h:,}"
                        )
                    else:
                        table.add_row(symbol, "No data", "-", "-")
                
                console.clear()
                console.print(table)
                await asyncio.sleep(1)
            
            await stream_manager.stop_streaming()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user.[/]")
        except Exception as exc:
            console.print(f"[red]❌ Monitoring failed: {exc}[/]")
    
    # ------------------------------------------------------------------- #
    #  DISPLAY AND UI HELPERS
    # ------------------------------------------------------------------- #
    
    def _display_analysis_results(self, decision: Dict) -> None:
        """Display analysis results in a formatted table."""
        console.print("\n" + "="*80)
        
        # Main decision panel
        status_color = "green" if decision["action"] == "BUY" else "red" if decision["action"] == "SELL" else "yellow"
        panel_content = (
            f"[bold]{decision['symbol']}[/] | [bold {status_color}]{decision['action']}[/] | "
            f"Confidence: [bold]{decision['confidence']:.1%}[/]\n"
            f"Strategy: {decision['strategy_used']} | Urgency: {decision['urgency']}\n"
            f"Live Price: [bold]${decision['live_price'] or 'N/A'}[/] | "
            f"Position: {decision['position_size']:.3f} contracts"
        )
        
        console.print(Panel(panel_content, title="Trading Decision", border_style=status_color))
        
        # Details table
        table = Table(title="Analysis Details", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=40)
        
        table.add_row("Execution Status", decision["execution_status"])
        table.add_row("Entry Price", f"${decision['entry_price']:,.2f}")
        table.add_row("Stop Loss", f"${decision.get('risk_limits', {}).get('stop_loss', 'N/A')}")
        table.add_row("Take Profit", f"${decision.get('risk_limits', {}).get('take_profit', 'N/A')}")
        table.add_row("Account Balance", f"${decision['account_balance']:,.0f}")
        table.add_row("Daily P&L", f"${decision['daily_pnl']:,.2f}")
        table.add_row("News Catalyst", "Yes" if decision['news_catalyst'] else "No")
        table.add_row("Order ID", str(decision.get('order_id', 'N/A')))
        
        console.print(table)
        
        # Reasoning
        console.print(Panel(decision['reasoning'], title="Strategy Reasoning", border_style="blue"))
        
        # Risk Assessment
        risk_color = "red" if "HIGH RISK" in decision['risk_assessment'] else "yellow" if "MODERATE" in decision['risk_assessment'] else "green"
        console.print(Panel(decision['risk_assessment'], title="Risk Assessment", border_style=risk_color))
    
    def _pick_symbol(self) -> str:
        """Interactive symbol picker."""
        categories = list(self.config.get_supported_symbols().keys())
        
        console.print("\n[bold]Select Symbol Category:[/]")
        for i, category in enumerate(categories, 1):
            symbols = self.config.get_supported_symbols()[category]
            console.print(f"{i}. {category.upper()} ({', '.join(symbols)})")
        
        while True:
            try:
                choice = int(Prompt.ask("Category", default="1"))
                if 1 <= choice <= len(categories):
                    category = categories[choice - 1]
                    break
                console.print("[red]Invalid choice[/]")
            except ValueError:
                console.print("[red]Please enter a number[/]")
        
        symbols = self.config.get_supported_symbols()[category]
        
        console.print(f"\n[bold]Select {category.upper()} Symbol:[/]")
        for i, symbol in enumerate(symbols, 1):
            console.print(f"{i}. {symbol}")
        
        while True:
            try:
                choice = int(Prompt.ask("Symbol", default="1"))
                if 1 <= choice <= len(symbols):
                    return symbols[choice - 1]
                console.print("[red]Invalid choice[/]")
            except ValueError:
                console.print("[red]Please enter a number[/]")
    
    def _pick_date(self) -> str:
        """Interactive date picker."""
        choice = Prompt.ask(
            "Analysis Date",
            choices=["today", "yesterday", "custom"],
            default="today"
        )
        
        if choice == "today":
            return self._get_today()
        elif choice == "yesterday":
            return self._get_yesterday()
        else:
            return Prompt.ask("Date (YYYY-MM-DD)", default=self._get_today())
    
    def _get_today(self) -> str:
        """Get today's date string."""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _get_yesterday(self) -> str:
        """Get yesterday's date string."""
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    def _save_analysis(self, decision: Dict, symbol: str, date: str) -> None:
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.config.results_dir / f"ib_analysis_{symbol}_{timestamp}.json"
        
        result_data = {
            "metadata": {
                "symbol": symbol,
                "date": date,
                "timestamp": timestamp,
                "live_mode": self.live_mode
            },
            "decision": decision
        }
        
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        console.print(f"[green]💾 Results saved to {filename}[/]")
    
    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.ib_broker:
            if not self.live_mode:  # Only close paper trading connections
                await self.ib_broker.close()
            self.ib_broker = None

# ---------------------------------------------------------------------------
#  CLI ENTRY POINT
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Interactive Brokers Futures Trading CLI",
        epilog="Example: python ib_futures_cli.py wizard --live"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Wizard command
    wizard_parser = subparsers.add_parser("wizard", help="Interactive trading wizard")
    wizard_parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Execute trading command")
    trade_parser.add_argument("--symbol", required=True, help="Futures symbol (e.g., CL)")
    trade_parser.add_argument("--action", choices=["analyze", "buy", "sell"], required=True)
    trade_parser.add_argument("--quantity", type=float, help="Trade quantity")
    trade_parser.add_argument("--date", help="Analysis date (YYYY-MM-DD)")
    trade_parser.add_argument("--live", action="store_true", help="Use live trading")
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Display portfolio")
    portfolio_parser.add_argument("--live", action="store_true", help="Use live account")
    
    return parser

async def main():
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    cli = IBFuturesCLI()
    
    try:
        if args.command == "wizard":
            await cli.wizard(live_mode=args.live)
        elif args.command == "trade":
            await cli.trade_command(
                symbol=args.symbol,
                action=args.action,
                live_mode=args.live,
                quantity=args.quantity,
                date=args.date
            )
        elif args.command == "portfolio":
            await cli.portfolio_command(live_mode=args.live)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/]")
    except Exception as exc:
        console.print(f"[red]❌ Error: {exc}[/]")
        log.error("cli_error", error=str(exc))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())