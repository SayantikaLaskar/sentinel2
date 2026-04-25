import json
import os
import time
from datetime import datetime
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

LOG_FILE = "../training_log.jsonl"

class DashboardState:
    def __init__(self):
        self.episodes = []
        self.thoughts = []
        self.services = {
            # Frontend
            "web-gateway": "healthy", "mobile-api": "healthy", "cdn-edge": "healthy",
            # App
            "cart-service": "healthy", "order-service": "healthy", "product-catalog": "healthy",
            "search-service": "healthy", "recommendation-engine": "healthy", "user-auth": "healthy",
            "notification-service": "healthy", "pricing-engine": "healthy", "inventory-service": "healthy",
            "review-service": "healthy", "wishlist-service": "healthy", "session-manager": "healthy",
            # Data
            "postgres-primary": "healthy", "postgres-replica": "healthy", "redis-cache": "healthy",
            "elasticsearch": "healthy", "kafka-broker": "healthy", "object-storage": "healthy",
            "analytics-db": "healthy", "audit-log": "healthy",
            # Infra
            "service-mesh": "healthy", "load-balancer": "healthy", "api-gateway": "healthy",
            "config-service": "healthy", "secret-manager": "healthy", "payment-vault": "healthy",
            "fraud-detector": "healthy"
        }
        self.mttr = 0
        self.reward = 0.0
        self.health = 100.0

    def update(self, line):
        if line.startswith("--- THOUGHT ---"):
            return # Start of thought block
        if line.startswith("---------------") or line.strip() == "":
            return # End/empty
        
        # Look for alert signals in thoughts to update the map
        # (Example: "[ALR] svc-05 -- cpu_spike=0.98")
        if "[ALR]" in line or "degraded" in line.lower():
            for svc in self.services:
                if svc in line:
                    self.services[svc] = "degraded"
        
        # Look for resolution signals
        if "CloseIncident" in line or "resolution" in line.lower():
            for svc in self.services:
                self.services[svc] = "healthy"

        try:
            data = json.loads(line)
            if "episode" in data:
                self.episodes.append(data)
                self.mttr = data.get("mttr", 0)
                self.reward = data.get("total_reward", 0.0)
                self.health = max(0, 100 + (self.reward * 10))
                # Reset map on episode end
                for svc in self.services:
                    self.services[svc] = "healthy"
        except:
            # It's a raw thought line
            ts = datetime.now().strftime("%H:%M:%S")
            self.thoughts.append(f"[{ts}] {line.strip()[:100]}")
            self.thoughts = self.thoughts[-15:]

def make_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(name="metrics", size=6),
        Layout(name="map"),
    )
    return layout

def get_header():
    return Panel(
        Text("SENTINEL CORE — NEURAL COMMAND CENTER", justify="center", style="bold cyan"),
        border_style="bright_blue"
    )

def get_footer(state):
    return Panel(
        Text(f"EPISODES: {len(state.episodes)} | MODE: LLM+GRPO | LOG: {LOG_FILE}", justify="center"),
        border_style="dim"
    )

def get_metrics(state):
    table = Table.grid(expand=True)
    table.add_column()
    table.add_column()
    
    health_color = "green" if state.health > 80 else "yellow" if state.health > 50 else "red"
    
    table.add_row(
        Panel(f"[bold {health_color}]{state.health:.1f}%[/]\nSLA HEALTH", border_style=health_color, expand=True),
        Panel(f"[bold yellow]{state.mttr}[/]\nCURRENT MTTR", border_style="yellow", expand=True)
    )
    table.add_row(
        Panel(f"[bold cyan]{state.reward:.3f}[/]\nLAST REWARD", border_style="cyan", expand=True),
        Panel(f"[bold magenta]{len(state.thoughts)}[/]\nTHOUGHT COUNT", border_style="magenta", expand=True)
    )
    return table

def get_map(state):
    cells = []
    for svc, status in state.services.items():
        color = "green" if status == "healthy" else "red"
        # Use a more compact label
        label = svc.replace("-service", "").replace("-gateway", "").replace("-engine", "")
        cells.append(Panel(f"[bold]{label}[/]", border_style=color, padding=(0,0)))
    
    return Panel(Columns(cells, equal=True, expand=True), title="NexaStack Service Topology", border_style="blue")

def get_thoughts(state):
    thought_text = "\n".join(state.thoughts)
    return Panel(thought_text, title="Neural Reasoning Stream", border_style="magenta")

def main():
    state = DashboardState()
    layout = make_layout()
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f: pass

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        
        with Live(layout, refresh_per_second=4, screen=True) as live:
            while True:
                line = f.readline()
                if line:
                    state.update(line)
                
                layout["header"].update(get_header())
                layout["footer"].update(get_footer(state))
                layout["metrics"].update(get_metrics(state))
                layout["map"].update(get_map(state))
                layout["right"].update(get_thoughts(state))
                time.sleep(0.1)

if __name__ == "__main__":
    main()
