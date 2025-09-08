#!/usr/bin/env python3
"""
Bet Copilot Mock Backend Demo

A simplified demonstration version that showcases the complete workflow
without external dependencies. This demonstrates the core business logic
and API structure for the complete Bet Copilot system.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Mock data structures to simulate API responses
mock_database = {
    "bets": [],
    "ledger": [
        {
            "entry_id": 1,
            "timestamp": datetime.now().isoformat(),
            "transaction_type": "Initial",
            "amount": 1000.0,
            "running_balance": 1000.0,
            "related_bet_id": None,
            "description": "Initial bankroll deposit"
        }
    ],
    "current_balance": 1000.0,
    "next_bet_id": 1,
    "next_entry_id": 2
}

def calculate_profit_loss(stake: float, odds: int, won: bool) -> float:
    """Calculate profit/loss for a bet based on American odds."""
    if not won:
        return -stake  # Full stake is lost
    
    if odds > 0:
        # Positive odds: profit = stake * (odds/100)
        return stake * (odds / 100)
    else:
        # Negative odds: profit = stake / (abs(odds)/100)
        return stake / (abs(odds) / 100)

def add_bet_demo(matchup: str, bet_type: str, stake: float, odds: int):
    """Demonstrate adding a new bet."""
    print(f"\n🎲 Adding New Bet:")
    print(f"   Matchup: {matchup}")
    print(f"   Type: {bet_type}")
    print(f"   Stake: ${stake:.2f}")
    print(f"   Odds: {odds:+d}")
    
    if mock_database["current_balance"] < stake:
        print(f"❌ Error: Insufficient funds! Current balance: ${mock_database['current_balance']:.2f}")
        return None
    
    # Create bet record
    bet = {
        "bet_id": mock_database["next_bet_id"],
        "matchup": matchup,
        "bet_type": bet_type,
        "stake": stake,
        "odds": odds,
        "status": "Pending",
        "profit_loss": 0.0,
        "bet_date": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Add to database
    mock_database["bets"].append(bet)
    
    # Update bankroll ledger
    new_balance = mock_database["current_balance"] - stake
    ledger_entry = {
        "entry_id": mock_database["next_entry_id"],
        "timestamp": datetime.now().isoformat(),
        "transaction_type": "Bet Placed",
        "amount": -stake,
        "running_balance": new_balance,
        "related_bet_id": bet["bet_id"],
        "description": f"Stake for bet #{bet['bet_id']}: {matchup}"
    }
    
    mock_database["ledger"].append(ledger_entry)
    mock_database["current_balance"] = new_balance
    mock_database["next_bet_id"] += 1
    mock_database["next_entry_id"] += 1
    
    print(f"✅ Bet #{bet['bet_id']} created successfully!")
    print(f"   New balance: ${new_balance:.2f}")
    
    return bet["bet_id"]

def settle_bet_demo(bet_id: int, result: str):
    """Demonstrate settling a bet."""
    print(f"\n🏁 Settling Bet #{bet_id} as {result}:")
    
    # Find bet
    bet = None
    for b in mock_database["bets"]:
        if b["bet_id"] == bet_id:
            bet = b
            break
    
    if not bet:
        print(f"❌ Error: Bet #{bet_id} not found!")
        return
    
    if bet["status"] != "Pending":
        print(f"❌ Error: Bet #{bet_id} is already settled as {bet['status']}!")
        return
    
    # Calculate profit/loss
    won = (result == "Won")
    profit_loss = calculate_profit_loss(bet["stake"], bet["odds"], won)
    
    # Update bet
    bet["status"] = result
    bet["profit_loss"] = profit_loss
    bet["updated_at"] = datetime.now().isoformat()
    
    print(f"   Original stake: ${bet['stake']:.2f}")
    print(f"   Odds: {bet['odds']:+d}")
    print(f"   Profit/Loss: ${profit_loss:+.2f}")
    
    # Update bankroll if won
    if won and profit_loss > 0:
        new_balance = mock_database["current_balance"] + profit_loss
        ledger_entry = {
            "entry_id": mock_database["next_entry_id"],
            "timestamp": datetime.now().isoformat(),
            "transaction_type": "Bet Settled",
            "amount": profit_loss,
            "running_balance": new_balance,
            "related_bet_id": bet_id,
            "description": f"Winnings for bet #{bet_id}: {bet['matchup']}"
        }
        
        mock_database["ledger"].append(ledger_entry)
        mock_database["current_balance"] = new_balance
        mock_database["next_entry_id"] += 1
        
        print(f"   New balance: ${new_balance:.2f}")
    
    print(f"✅ Bet #{bet_id} settled successfully!")

def get_dashboard_stats_demo():
    """Demonstrate dashboard statistics calculation."""
    print(f"\n📊 Dashboard Statistics:")
    
    # Calculate total P/L
    total_pl = sum(bet["profit_loss"] for bet in mock_database["bets"])
    
    # Calculate total stakes for ROI
    total_stakes = sum(bet["stake"] for bet in mock_database["bets"])
    roi = (total_pl / total_stakes * 100) if total_stakes > 0 else 0.0
    
    # Calculate win rate
    settled_bets = [bet for bet in mock_database["bets"] if bet["status"] in ["Won", "Lost"]]
    won_bets = [bet for bet in settled_bets if bet["status"] == "Won"]
    win_rate = (len(won_bets) / len(settled_bets) * 100) if settled_bets else 0.0
    
    # Count bets
    total_bets = len(mock_database["bets"])
    pending_bets = len([bet for bet in mock_database["bets"] if bet["status"] == "Pending"])
    
    print(f"   Current Balance: ${mock_database['current_balance']:.2f}")
    print(f"   Total P/L: ${total_pl:+.2f}")
    print(f"   ROI: {roi:+.1f}%")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Total Bets: {total_bets}")
    print(f"   Pending Bets: {pending_bets}")
    
    return {
        "current_balance": mock_database["current_balance"],
        "total_profit_loss": total_pl,
        "roi": roi,
        "win_rate": win_rate,
        "total_bets": total_bets,
        "pending_bets": pending_bets,
        "recent_bets": mock_database["bets"][-20:]  # Last 20 bets
    }

def demo_complete_workflow():
    """Demonstrate the complete Bet Copilot workflow."""
    print("🚀 Bet Copilot Alpha Demo - Complete Workflow")
    print("=" * 60)
    
    # Initial state
    print(f"\n💰 Starting bankroll: ${mock_database['current_balance']:.2f}")
    
    # Add some bets
    bet1_id = add_bet_demo("Lakers vs Warriors", "Moneyline", 100.0, 150)
    bet2_id = add_bet_demo("Chiefs vs Bills", "Point Spread", 75.0, -110)
    bet3_id = add_bet_demo("Over 45.5 Points", "Over/Under", 50.0, -105)
    
    # Show dashboard after bets placed
    get_dashboard_stats_demo()
    
    # Settle some bets
    settle_bet_demo(bet1_id, "Won")  # Lakers win!
    settle_bet_demo(bet2_id, "Lost")  # Chiefs lose
    # Leave bet3 pending
    
    # Final dashboard
    print(f"\n" + "=" * 60)
    print("🏁 Final Results:")
    final_stats = get_dashboard_stats_demo()
    
    # Show bet history
    print(f"\n📜 Bet History:")
    for bet in mock_database["bets"]:
        status_emoji = {"Pending": "⏳", "Won": "🏆", "Lost": "❌"}[bet["status"]]
        print(f"   {status_emoji} #{bet['bet_id']}: {bet['matchup']} - ${bet['stake']:.2f} @ {bet['odds']:+d} = ${bet['profit_loss']:+.2f}")
    
    # Show ledger
    print(f"\n💳 Transaction Ledger:")
    for entry in mock_database["ledger"]:
        print(f"   {entry['transaction_type']}: ${entry['amount']:+.2f} (Balance: ${entry['running_balance']:.2f})")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🎯 This demonstrates the complete Bet Copilot system architecture and business logic.")

if __name__ == "__main__":
    demo_complete_workflow()