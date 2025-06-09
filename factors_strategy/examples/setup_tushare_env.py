#!/usr/bin/env python
"""
Script to help set up and verify Tushare environment configuration
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def check_tushare_token():
    """Check if TUSHARE_TOKEN environment variable is set"""
    token = os.environ.get('TUSHARE_TOKEN')
    
    if token:
        # Mask the token for security (show only first and last 4 characters)
        masked_token = f"{token[:4]}{'*' * (len(token) - 8)}{token[-4:]}" if len(token) > 8 else "*" * len(token)
        print(f"✅ TUSHARE_TOKEN is set: {masked_token}")
        return True
    else:
        print("❌ TUSHARE_TOKEN environment variable is not set")
        return False


def set_tushare_token_instructions():
    """Print instructions for setting TUSHARE_TOKEN"""
    print("\n📝 How to set TUSHARE_TOKEN environment variable:\n")
    
    print("1. Get your Tushare API token from: https://tushare.pro/user/token")
    print("\n2. Set the environment variable:\n")
    
    # Detect OS
    if sys.platform.startswith('win'):
        print("   Windows (Command Prompt):")
        print("   > set TUSHARE_TOKEN=your_token_here")
        print("\n   Windows (PowerShell):")
        print("   > $env:TUSHARE_TOKEN=\"your_token_here\"")
    else:
        print("   Linux/Mac (temporary - current session only):")
        print("   $ export TUSHARE_TOKEN=\"your_token_here\"")
        print("\n   Linux/Mac (permanent - add to ~/.bashrc or ~/.zshrc):")
        print("   $ echo 'export TUSHARE_TOKEN=\"your_token_here\"' >> ~/.bashrc")
        print("   $ source ~/.bashrc")
    
    print("\n3. Verify it's set by running this script again")


def test_tushare_connection():
    """Test if Tushare can be initialized with the token"""
    token = os.environ.get('TUSHARE_TOKEN')
    
    if not token:
        print("\n⚠️  Cannot test Tushare connection - TUSHARE_TOKEN not set")
        return False
    
    try:
        import tushare as ts
        print("\n🔄 Testing Tushare connection...")
        
        # Set token and create API instance
        ts.set_token(token)
        pro = ts.pro_api()
        
        # Try to get stock basic info (minimal data request)
        df = pro.stock_basic(exchange='', list_status='L', limit=1)
        
        if not df.empty:
            print("✅ Tushare connection successful!")
            print(f"   Found {len(df)} test record(s)")
            return True
        else:
            print("⚠️  Tushare connected but returned no data")
            return False
            
    except ImportError:
        print("❌ Tushare package not installed. Run: pip install tushare")
        return False
    except Exception as e:
        print(f"❌ Tushare connection failed: {e}")
        return False


def main():
    """Main function"""
    print("🔧 Tushare Environment Setup Checker")
    print("=" * 50)
    
    # Check if token is set
    token_set = check_tushare_token()
    
    if not token_set:
        set_tushare_token_instructions()
    else:
        # Test connection if token is set
        test_tushare_connection()
    
    print("\n" + "=" * 50)
    print("📌 Remember: Never commit your API token to version control!")
    print("   Always use environment variables for sensitive data.")


if __name__ == "__main__":
    main()