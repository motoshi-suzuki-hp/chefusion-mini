#!/usr/bin/env python3
"""
Demo script to test log monitoring functionality.

This script generates various log entries and demonstrates the log monitoring
capabilities of the fusion cuisine application.
"""

import logging
import time
import random
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import create_log_monitor, setup_logging


def simulate_application_logs():
    """Simulate various application log scenarios."""
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger("fusion_cuisine_demo")
    
    # Log various scenarios
    scenarios = [
        ("Starting fusion cuisine application", "info"),
        ("Loading recipe data", "info"),
        ("Processing 1000 recipes", "info"),
        ("Model training took 45.2 seconds", "info"),
        ("Memory usage: 1.2 GB", "info"),
        ("Warning: Low disk space", "warning"),
        ("GPU memory usage: 6.8 GB", "info"),
        ("Failed to load image: corrupted file", "error"),
        ("Network connection timeout", "error"),
        ("Recipe generation completed in 12.3 seconds", "info"),
        ("Warning: High CPU usage detected", "warning"),
        ("CUDA out of memory error", "error"),
        ("Model evaluation took 8.7 seconds", "info"),
        ("Fusion recipe created successfully", "info"),
        ("Warning: API rate limit approaching", "warning"),
        ("Critical: Database connection lost", "critical"),
        ("System recovered successfully", "info"),
        ("Performance metrics: 95% accuracy", "info"),
        ("Memory usage: 2.8 GB", "info"),
        ("Processing time: 180 seconds", "info"),
    ]
    
    print("🎬 Starting log simulation...")
    print("📝 Generating various log entries...")
    
    for i, (message, level) in enumerate(scenarios, 1):
        # Add some randomness to timing
        time.sleep(random.uniform(0.5, 2.0))
        
        # Log the message
        if level == "info":
            logger.info(f"[Step {i}/20] {message}")
        elif level == "warning":
            logger.warning(f"[Step {i}/20] {message}")
        elif level == "error":
            logger.error(f"[Step {i}/20] {message}")
        elif level == "critical":
            logger.critical(f"[Step {i}/20] {message}")
        
        print(f"  📊 Generated {level.upper()}: {message}")
        
        # Add some extra errors to test rate limiting
        if i == 10:
            print("  🚨 Simulating error burst...")
            for j in range(5):
                logger.error(f"Burst error {j+1}: Connection failed")
                time.sleep(0.1)
    
    print("✅ Log simulation complete!")


def demonstrate_monitoring():
    """Demonstrate the log monitoring capabilities."""
    print("\n🔍 Demonstrating Log Monitoring")
    print("=" * 50)
    
    # Create monitor
    monitor = create_log_monitor()
    
    # Add custom alert handler
    def demo_alert_handler(message: str):
        print(f"\n🚨 DEMO ALERT: {message}")
    
    monitor.add_alert_handler(demo_alert_handler)
    
    # Start monitoring
    print("🎯 Starting log monitor...")
    monitor.start_monitoring()
    
    # Let it run for a bit to analyze existing logs
    print("📊 Analyzing existing logs...")
    time.sleep(2)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Show statistics
    print("\n📈 Log Analysis Results:")
    monitor.print_stats()
    
    # Show recent errors
    recent_errors = monitor.get_recent_errors(3)
    if recent_errors:
        print("🔴 Recent Errors:")
        for error in recent_errors:
            print(f"  • {error}")
    
    # Export statistics
    stats_file = "logs/demo_monitoring_stats.json"
    monitor.export_stats(stats_file)
    print(f"\n📄 Statistics exported to {stats_file}")


def main():
    """Main demonstration function."""
    print("🎯 Fusion Cuisine Log Monitoring Demo")
    print("=" * 50)
    
    # First, simulate some application logs
    simulate_application_logs()
    
    # Then demonstrate monitoring
    demonstrate_monitoring()
    
    print("\n✅ Demo completed successfully!")
    print("💡 To run continuous monitoring, use: make monitor-logs")
    print("💡 To view JupyterLab interface, use: make jupyter")


if __name__ == "__main__":
    main()