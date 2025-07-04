"""scalping_cli.py – Ultra-fast scalping CLI for futures trading.

Real-time scalping interface with nanosecond-aware execution:
• Tick-by-tick analysis and execution
• Sub-second signal generation
• Real-time P&L monitoring
• Risk controls and circuit breakers
• Live order book analysis
• News reaction scalping

Usage:
    python cli/scalping_cli.py --symbol ES --timeframe 1m --live
    python cli/scalping_cli.py monitor --symbols ES,CL,GC
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import structlog
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.strategies.scalping_strategies import create_scalping_engine, TF
from tradingagents.brokers.ib_broker import create_ib_broker
from tradingagents.brokers.ib_realtime_adapter import create_ib_stream_manager
from tradingagents.default_config import DEFAULT_CONFIG

console = Console()
log = structlog.get_logger("scalping_cli")

class ScalpingCLI:
    """Ultra-fast scalping CLI interface."""
    
    def __init__(self):
        self.scalping_engine = None
        self.ib_broker = None
        self.stream_manager = None
        self.live_mode = False
        self.running = False
        self.metrics = {}
        
    async def initialize(self, symbols: List[str], timeframe: str, live_mode: bool):
        """Initialize scalping components."""
        self.live_mode = live_mode
        
        console.print(Panel.fit(
            f"⚡ [bold blue]SCALPING ENGINE[/bold blue] ⚡\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Timeframe: {timeframe}\n"
            f"Mode: [{'red' if live_mode else 'green'}]{'LIVE' if live_mode else 'PAPER'}[/]\n"
            f"{'[red]⚠️  REAL MONEY - ULTRA FAST EXECUTION ⚠️[/]' if live_mode else '[green]✅ Safe simulation mode[/]'}",
            title="Futures Scalping System"
        ))
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Initialize IB broker
                task1 = progress.add_task("Connecting to IB...", total=None)
                self.ib_broker = await create_ib_broker(paper_trading=not live_mode)
                progress.update(task1, description="✅ IB Connected")
                
                # Initialize streaming
                task2 = progress.add_task("Setting up real-time data...", total=None)
                self.stream_manager = await create_ib_stream_manager(symbols, self.ib_broker)
                progress.update(task2, description="✅ Streaming Ready")
                
                # Initialize scalping engine
                task3 = progress.add_task("Loading scalping engine...", total=None)
                tf_enum = getattr(TF, timeframe.upper().replace('M', 'M').replace('S', 'S'))
                self.scalping_engine = await create_scalping_engine(symbols, tf_enum)
                progress.update(task3, description="✅ Scalping Engine Ready")
            
            console.print("[green]🚀 All systems ready for scalping![/]")
            
        except Exception as exc:
            console.print(f"[red]❌ Initialization failed: {exc}[/]")
            raise
    
    async def start_scalping(self, symbols: List[str], duration_minutes: int = 60):
        """Start scalping session."""
        console.print(f"\n[yellow]🎯 Starting {duration_minutes}-minute scalping session...[/]")
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Subscribe to real-time data
        for symbol in symbols:
            def make_tick_handler(sym):
                async def on_tick(symbol_data, tick_data):
                    if self.running and symbol_data == sym:
                        signal = await self.scalping_engine.on_tick(
                            sym, tick_data.price, tick_data.volume, tick_data.timestamp
                        )
                        if signal and self.live_mode:
                            # Execute signal through IB
                            await self._execute_scalp_signal(signal)
                return on_tick
            
            self.stream_manager.subscribe("tick", make_tick_handler(symbol))
        
        # Main scalping loop
        signal_count = 0
        try:
            while self.running and datetime.now() < end_time:
                await asyncio.sleep(0.1)  # 100ms refresh
                
                # Update metrics
                self.metrics = self.scalping_engine.get_metrics()
                
                # Check for new signals (this would be triggered by ticks)
                # In real implementation, signals come from tick handlers
                
                # Display live updates every 5 seconds
                if int(datetime.now().second) % 5 == 0:
                    await self._display_live_metrics(symbols, signal_count)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️  Scalping session stopped by user[/]")
        finally:
            self.running = False
            await self._display_session_summary(signal_count, start_time)
    
    async def _execute_scalp_signal(self, signal):
        """Execute scalping signal through IB."""
        if not self.ib_broker:
            return
            
        try:
            trade = await self.ib_broker.execute_signal(signal)
            if trade:
                console.print(f"[green]⚡ EXECUTED: {signal.action} {signal.symbol} @ ${signal.entry_price:.4f}[/]")
            else:
                console.print(f"[red]❌ REJECTED: {signal.symbol} - Risk limits[/]")
        except Exception as exc:
            console.print(f"[red]❌ EXECUTION ERROR: {exc}[/]")
    
    async def _display_live_metrics(self, symbols: List[str], signal_count: int):
        """Display live scalping metrics."""
        # Create metrics table
        table = Table(title="📊 Live Scalping Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Signals Generated", str(signal_count))
        table.add_row("Daily P&L", f"${self.metrics.get('daily_pnl', 0):,.2f}")
        table.add_row("Risk Utilization", f"{self.metrics.get('risk_utilization', {}).get('daily_loss_pct', 0):.1f}%")
        table.add_row("Active Symbols", str(len(symbols)))
        
        # Portfolio summary
        if self.ib_broker:
            portfolio = self.ib_broker.get_portfolio_summary()
            table.add_row("Account P&L", f"${portfolio.get('daily_pnl', 0):,.2f}")
            table.add_row("Open Positions", str(portfolio.get('total_positions', 0)))
        
        console.clear()
        console.print(table)
    
    async def _display_session_summary(self, signal_count: int, start_time: datetime):
        """Display final session summary."""
        duration = datetime.now() - start_time
        
        console.print("\n" + "="*60)
        console.print("[bold blue]📈 SCALPING SESSION COMPLETE[/bold blue]")
        console.print("="*60)
        console.print(f"Duration: {duration}")
        console.print(f"Signals Generated: {signal_count}")
        console.print(f"Signals per Hour: {signal_count / max(duration.seconds / 3600, 1):.1f}")
        
        if self.metrics:
            console.print(f"Final P&L: ${self.metrics.get('daily_pnl', 0):,.2f}")
            risk_util = self.metrics.get('risk_utilization', {})
            console.print(f"Risk Utilization: {risk_util.get('daily_loss_pct', 0):.1f}%")
        
        console.print("="*60)
    
    async def monitor_mode(self, symbols: List[str]):
        """Real-time monitoring mode."""
        console.print(f"[green]📊 Monitoring {', '.join(symbols)} in real-time[/]")
        console.print("[dim]Press Ctrl+C to stop...[/dim]")
        
        try:
            while True:
                # Create live data table
                table = Table(title="🔴 LIVE MARKET DATA", show_header=True)
                table.add_column("Symbol", style="cyan")
                table.add_column("Price", style="white")
                table.add_column("Change", style="white")
                table.add_column("Volume", style="white")
                table.add_column("Spread", style="white")
                table.add_column("Signal", style="white")
                
                for symbol in symbols:
                    if self.stream_manager:
                        summary = self.stream_manager.get_market_summary(symbol)
                        liquidity = self.stream_manager.get_liquidity_metrics(symbol)
                        
                        if summary:
                            change_color = "green" if summary.change_24h >= 0 else "red"
                            table.add_row(
                                symbol,
                                f"${summary.last_price:.4f}",
                                f"[{change_color}]{summary.change_24h:+.2f}%[/]",
                                f"{summary.volume_24h:,}",
                                f"{liquidity.get('spread_bps', 0):.1f}bp",
                                "🟢" if abs(summary.change_24h) > 0.1 else "⚪"
                            )
                        else:
                            table.add_row(symbol, "No data", "-", "-", "-", "⚪")
                    else:
                        table.add_row(symbol, "No stream", "-", "-", "-", "⚪")
                
                # Update display
                console.clear()
                console.print(table)
                
                await asyncio.sleep(1)  # Update every second
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/]")
    
    async def cleanup(self):
        """Cleanup resources."""
        self.running = False
        
        if self.scalping_engine:
            await self.scalping_engine.close()
        
        if self.stream_manager:
            await self.stream_manager.stop_streaming()
        
        if self.ib_broker and not self.live_mode:
            await self.ib_broker.close()

# ---------------------------------------------------------------------------
#  CLI INTERFACE
# ---------------------------------------------------------------------------

async def scalp_command(args):
    """Execute scalping command."""
    cli = ScalpingCLI()
    symbols = [args.symbol.upper()]
    
    try:
        await cli.initialize(symbols, args.timeframe, args.live)
        await cli.start_scalping(symbols, args.duration)
    finally:
        await cli.cleanup()

async def monitor_command(args):
    """Execute monitoring command."""
    cli = ScalpingCLI()
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    try:
        await cli.initialize(symbols, args.timeframe, args.live)
        await cli.monitor_mode(symbols)
    finally:
        await cli.cleanup()

def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Ultra-Fast Futures Scalping CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Scalp command
    scalp_parser = subparsers.add_parser("scalp", help="Start scalping session")
    scalp_parser.add_argument("--symbol", required=True, help="Symbol to scalp (e.g., ES)")
    scalp_parser.add_argument("--timeframe", choices=["tick", "5s", "15s", "1m", "5m"], default="1m")
    scalp_parser.add_argument("--duration", type=int, default=60, help="Session duration in minutes")
    scalp_parser.add_argument("--live", action="store_true", help="Use live trading")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Real-time monitoring")
    monitor_parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., ES,CL,GC)")
    monitor_parser.add_argument("--timeframe", choices=["tick", "5s", "15s", "1m", "5m"], default="1m")
    monitor_parser.add_argument("--live", action="store_true", help="Use live data")
    
    return parser

async def main():
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    # Verify environment
    if args.live:
        ib_key = os.getenv("IB_HOST")
        if not ib_key:
            console.print("[red]❌ IB_HOST not configured for live trading[/]")
            return
    
    try:
        if args.command == "scalp":
            await scalp_command(args)
        elif args.command == "monitor":
            await monitor_command(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/]")
    except Exception as exc:
        console.print(f"[red]❌ Error: {exc}[/]")
        log.error("cli_error", error=str(exc))

if __name__ == "__main__":
    asyncio.run(main())