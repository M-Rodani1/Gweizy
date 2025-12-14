#!/usr/bin/env python3
"""
Safe training wrapper that checks prerequisites before training
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_train():
    """Train models only if database has enough data"""
    try:
        print("Checking database for sufficient training data...")
        from data.database import DatabaseManager

        db = DatabaseManager()
        historical = db.get_historical_data(hours=168)  # 7 days

        print(f"Found {len(historical)} historical records")

        if len(historical) < 100:
            print(f"⚠️  Not enough data ({len(historical)} records). Need at least 100.")
            print("   Skipping model training. API will use fallback predictions.")
            return False

        print(f"✅ Sufficient data found. Starting model training...")

        # Import and run training
        from scripts.train_directional_optimized import train_directional_optimized
        train_directional_optimized()

        print("✅ Model training completed successfully!")
        return True

    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("   Dependencies may not be installed correctly.")
        return False

    except Exception as e:
        print(f"⚠️  Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   API will use fallback predictions.")
        return False

if __name__ == '__main__':
    success = safe_train()
    sys.exit(0 if success else 1)
