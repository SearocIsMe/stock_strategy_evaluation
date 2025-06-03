#!/usr/bin/env python
"""
Test script to verify system setup and components
"""

import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
import xgboost
import lightgbm
import clickhouse_driver
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{text.center(60)}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def test_python_version():
    """Test Python version"""
    print_header("Python Version Check")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - Requires 3.10+")
        return False
    return True


def test_imports():
    """Test critical imports"""
    print_header("Import Check")
    
    modules = [
        ('numpy', np.__version__),
        ('pandas', pd.__version__),
        ('torch', torch.__version__),
        ('xgboost', xgboost.__version__),
        ('lightgbm', lightgbm.__version__),
        ('clickhouse_driver', clickhouse_driver.__version__),
    ]
    
    all_good = True
    for module_name, version in modules:
        try:
            print_success(f"{module_name} {version}")
            
            # Special check for PyTorch version
            if module_name == 'torch':
                torch_version_clean = version.split('+')[0]  # Remove +cu124 suffix
                torch_major, torch_minor = torch_version_clean.split('.')[:2]
                if int(torch_major) < 2 or (int(torch_major) == 2 and int(torch_minor) < 0):
                    print_error(f"  PyTorch version {version} is too old. Requires 2.0.0+")
                    all_good = False
                    
        except Exception as e:
            print_error(f"{module_name} - {str(e)}")
            all_good = False
            
    return all_good


def test_gpu():
    """Test GPU availability"""
    print_header("GPU Check")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_success(f"CUDA available - {gpu_count} GPU(s) detected")
        print_success(f"CUDA version: {torch.version.cuda}")
        print_success(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpu_capability = torch.cuda.get_device_capability(i)
            print_success(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB, Compute Capability: {gpu_capability[0]}.{gpu_capability[1]})")
            
            # Check if it's an H100 or similar high-end GPU
            if "H100" in gpu_name or "A100" in gpu_name:
                print_success(f"    High-performance GPU detected! Excellent for deep learning.")
                
        return True
    else:
        print_warning("No CUDA GPU detected - will use CPU")
        print_warning("For optimal performance, consider using a GPU (especially NVIDIA H100)")
        return False


def test_directories():
    """Test directory structure"""
    print_header("Directory Structure Check")
    
    required_dirs = [
        'config',
        'data',
        'data/storage',
        'factors',
        'factors/traditional',
        'factors/ai_generated',
        'models',
        'models/deep_learning',
        'models/ensemble',
        'strategy',
        'infrastructure',
        'infrastructure/deploy',
        'infrastructure/backup',
        'visualization',
        'docs',
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"{dir_path}/")
        else:
            print_error(f"{dir_path}/ - Missing")
            all_good = False
            
    return all_good


def test_config_files():
    """Test configuration files"""
    print_header("Configuration Files Check")
    
    config_files = [
        'config/database.yaml',
        'config/model.yaml',
        'config/factors.yaml',
        'config/strategy.yaml',
    ]
    
    all_good = True
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
                print_success(f"{config_file} - Valid")
            except Exception as e:
                print_error(f"{config_file} - Invalid YAML: {str(e)}")
                all_good = False
        else:
            print_error(f"{config_file} - Missing")
            all_good = False
            
    return all_good


def test_clickhouse_connection():
    """Test ClickHouse connection"""
    print_header("ClickHouse Connection Check")
    
    try:
        # Load database config
        with open('config/database.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        db_config = config['database']['clickhouse']
        
        # Try to connect
        from clickhouse_driver import Client
        client = Client(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        # Test query
        result = client.execute('SELECT version()')
        version = result[0][0]
        print_success(f"Connected to ClickHouse {version}")
        
        # Check database
        result = client.execute(f"EXISTS DATABASE {db_config['database']}")
        if result[0][0]:
            print_success(f"Database '{db_config['database']}' exists")
        else:
            print_warning(f"Database '{db_config['database']}' does not exist - run initialize.py")
            
        return True
        
    except Exception as e:
        print_error(f"ClickHouse connection failed: {str(e)}")
        print_warning("Make sure ClickHouse is running (docker-compose up -d)")
        return False


def test_sample_factor_calculation():
    """Test sample factor calculation"""
    print_header("Factor Calculation Test")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': '000001.SZ',
            'price': 10 + np.random.randn(100) * 0.1,
            'volume': np.random.randint(1000, 10000, 100),
            'bid_price': 10 + np.random.randn(100) * 0.1 - 0.01,
            'ask_price': 10 + np.random.randn(100) * 0.1 + 0.01,
        })
        
        # Calculate simple factors
        sample_data['returns'] = sample_data['price'].pct_change()
        sample_data['spread'] = sample_data['ask_price'] - sample_data['bid_price']
        sample_data['mid_price'] = (sample_data['ask_price'] + sample_data['bid_price']) / 2
        
        print_success(f"Generated {len(sample_data)} sample tick records")
        print_success(f"Calculated {len(sample_data.columns) - 5} factors")
        
        return True
        
    except Exception as e:
        print_error(f"Factor calculation failed: {str(e)}")
        return False


def test_model_inference():
    """Test model inference"""
    print_header("Model Inference Test")
    
    try:
        # Create sample features
        n_samples = 10
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        
        # Test XGBoost
        xgb_model = xgboost.XGBClassifier(n_estimators=10, max_depth=3)
        y = np.random.randint(0, 2, n_samples)
        xgb_model.fit(X, y)
        predictions = xgb_model.predict_proba(X)
        print_success(f"XGBoost inference: {predictions.shape}")
        
        # Test PyTorch
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        tensor_X = torch.FloatTensor(X).to(device)
        simple_model = torch.nn.Linear(n_features, 1).to(device)
        with torch.no_grad():
            output = torch.sigmoid(simple_model(tensor_X))
        print_success(f"PyTorch inference on {device}: {output.shape}")
        
        # Test mixed precision if GPU available
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output_fp16 = torch.sigmoid(simple_model(tensor_X))
            print_success(f"Mixed precision (FP16) inference: {output_fp16.shape}")
        
        return True
        
    except Exception as e:
        print_error(f"Model inference failed: {str(e)}")
        return False


def test_library_compatibility():
    """Test library compatibility"""
    print_header("Library Compatibility Check")
    
    try:
        # Test pandas with PyArrow backend
        df = pd.DataFrame({'a': [1, 2, 3]})
        df_arrow = df.convert_dtypes(dtype_backend='pyarrow')
        print_success("Pandas with PyArrow backend: OK")
        
        # Test async functionality
        import asyncio
        async def test_async():
            return True
        
        result = asyncio.run(test_async())
        print_success("Async support: OK")
        
        # Test Numba JIT compilation
        from numba import jit
        @jit(nopython=True)
        def test_numba(x):
            return x * 2
        
        result = test_numba(5)
        print_success("Numba JIT compilation: OK")
        
        return True
        
    except Exception as e:
        print_error(f"Compatibility test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print_header("Stock Strategy System Test")
    print("Testing system components and configuration...\n")
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("Directory Structure", test_directories),
        ("Configuration Files", test_config_files),
        ("ClickHouse Connection", test_clickhouse_connection),
        ("Factor Calculation", test_sample_factor_calculation),
        ("Model Inference", test_model_inference),
        ("Library Compatibility", test_library_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"{test_name} - Unexpected error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    # Version summary
    print("\n" + "="*60)
    print("Version Summary:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print("="*60)
    
    if passed == total:
        print_success("\nAll tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Initialize database: python -m data.storage.initialize")
        print("2. Run strategy: python run_strategy.py")
        print("3. Start dashboard: python run_strategy.py --mode dashboard")
    else:
        print_error(f"\n{total - passed} tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Start services: ./infrastructure/deploy/setup.sh")
        print("- Check configuration files in config/")
        print("- Ensure Python 3.10+ is installed")
        print("- Ensure PyTorch 2.0.0+ is installed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)