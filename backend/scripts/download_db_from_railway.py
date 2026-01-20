#!/usr/bin/env python3
"""
Download gas_data.db from Railway volume.

This script helps you download the database file from Railway
so you can upload it to Google Colab for training.

Usage:
    python download_db_from_railway.py

Requirements:
    - Railway CLI installed: npm i -g @railway/cli
    - Logged in: railway login
    - Project linked: railway link
"""

import os
import subprocess
import sys

def check_railway_cli():
    """Check if Railway CLI is installed"""
    try:
        result = subprocess.run(['railway', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_database():
    """Download database from Railway"""
    print("📥 Downloading database from Railway...\n")
    
    # Check if Railway CLI is installed
    if not check_railway_cli():
        print("❌ Railway CLI not found!")
        print("\nInstall it with:")
        print("  npm i -g @railway/cli")
        print("\nAlternative methods:")
        print("\n1. Railway Dashboard:")
        print("   - Go to https://railway.app")
        print("   - Select your project → Service → Volumes")
        print("   - Navigate to backend/gas_data.db and download")
        print("\n2. SSH into Railway service:")
        print("   - railway shell")
        print("   - cat backend/gas_data.db > /tmp/gas_data.db")
        print("   - Download via Railway dashboard file browser")
        return False
    
    # Check if logged in
    try:
        subprocess.run(['railway', 'whoami'], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Not logged in to Railway")
        print("Please run: railway login")
        return False
    
    # Download the database
    print("Downloading backend/gas_data.db...")
    try:
        with open('gas_data.db', 'wb') as f:
            result = subprocess.run(
                ['railway', 'run', 'cat', 'backend/gas_data.db'],
                stdout=f,
                stderr=subprocess.PIPE,
                check=True
            )
        
        if os.path.exists('gas_data.db'):
            file_size_mb = os.path.getsize('gas_data.db') / (1024 * 1024)
            print(f"\n✅ Successfully downloaded gas_data.db")
            print(f"   File size: {file_size_mb:.2f} MB")
            print(f"   Location: {os.path.abspath('gas_data.db')}")
            print("\n📋 Next steps:")
            print("   1. Upload this file to Google Colab")
            print("   2. Or use it locally for training")
            return True
        else:
            print("\n❌ Download failed - file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to download database")
        print(f"   Error: {e.stderr.decode() if e.stderr else str(e)}")
        print("\n💡 Troubleshooting:")
        print("   - Make sure project is linked: railway link")
        print("   - Check that volume is mounted correctly")
        print("   - Verify file path: backend/gas_data.db")
        return False

if __name__ == '__main__':
    success = download_database()
    sys.exit(0 if success else 1)
