#!/usr/bin/env python3
"""
Log monitoring script for the fusion cuisine application.

This script provides a standalone log monitoring tool that can be run
alongside the main application to monitor server logs in real-time.

Usage:
    python scripts/monitor_logs.py [--log-file LOG_FILE] [--error-threshold N] [--warning-threshold N]

Example:
    python scripts/monitor_logs.py --log-file logs/fusion_cuisine.log --error-threshold 5
"""

import argparse
import sys
import time
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import create_log_monitor, setup_logging


def main():
    """Main function for the log monitoring script."""
    parser = argparse.ArgumentParser(description='Monitor fusion cuisine application logs')
    parser.add_argument('--log-file', default='logs/fusion_cuisine.log',
                        help='Path to the log file to monitor (default: logs/fusion_cuisine.log)')
    parser.add_argument('--error-threshold', type=int, default=10,
                        help='Number of errors per minute to trigger alert (default: 10)')
    parser.add_argument('--warning-threshold', type=int, default=20,
                        help='Number of warnings per minute to trigger alert (default: 20)')
    parser.add_argument('--stats-interval', type=int, default=30,
                        help='Interval in seconds between statistics updates (default: 30)')
    parser.add_argument('--export-stats', action='store_true',
                        help='Export final statistics to JSON file')
    
    args = parser.parse_args()
    
    # Setup logging for the monitor script
    setup_logging("INFO")
    
    print(f"🔍 Starting log monitoring for: {args.log_file}")
    print(f"📊 Error threshold: {args.error_threshold} errors/minute")
    print(f"⚠️  Warning threshold: {args.warning_threshold} warnings/minute")
    print(f"📈 Stats interval: {args.stats_interval} seconds")
    print("-" * 60)
    
    # Create and configure the log monitor
    monitor = create_log_monitor(args.log_file)
    monitor.error_threshold = args.error_threshold
    monitor.warning_threshold = args.warning_threshold
    
    # Add custom alert handler for email or webhook notifications
    def custom_alert_handler(message: str):
        """Custom alert handler - extend this for email/webhook notifications."""
        # For now, just print with emphasis
        print(f"\n{'='*60}")
        print(f"🚨 ALERT: {message}")
        print(f"{'='*60}\n")
        
        # Here you could add:
        # - Email notifications
        # - Webhook calls
        # - Slack notifications
        # - Log to external monitoring systems
    
    monitor.add_alert_handler(custom_alert_handler)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        print("✅ Log monitoring active. Press Ctrl+C to stop.")
        print("📊 Statistics will be updated every {} seconds".format(args.stats_interval))
        print()
        
        # Main monitoring loop
        while True:
            time.sleep(args.stats_interval)
            monitor.print_stats()
            
            # Show recent errors if any
            recent_errors = monitor.get_recent_errors(5)
            if recent_errors:
                print("🔴 Recent Errors:")
                for error in recent_errors:
                    print(f"  {error}")
                print()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping log monitoring...")
        monitor.stop_monitoring()
        
        print("\n📊 Final Statistics:")
        monitor.print_stats()
        
        # Export statistics if requested
        if args.export_stats:
            stats_file = "logs/monitoring_stats.json"
            monitor.export_stats(stats_file)
            print(f"📄 Statistics exported to {stats_file}")
        
        print("✅ Log monitoring stopped.")
    
    except Exception as e:
        print(f"❌ Error in log monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()