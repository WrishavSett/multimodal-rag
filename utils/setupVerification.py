#!/usr/bin/env python3
"""
Setup Verification Script for Multimodal RAG System
Verifies that all dependencies and system requirements are properly installed
"""

import sys
import subprocess
import os
from pathlib import Path
import importlib

def print_status(message, status="INFO"):
    """Print formatted status message"""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m", 
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "END": "\033[0m"
    }
    print(f"{colors.get(status, colors['INFO'])}|{status}| {message}{colors['END']}")

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires 3.9+", "ERROR")
        return False

def check_package_installation(packages):
    """Check if required packages are installed"""
    results = {}
    for package in packages:
        try:
            importlib.import_module(package)
            print_status(f"Package '{package}' - Installed", "SUCCESS")
            results[package] = True
        except ImportError:
            print_status(f"Package '{package}' - Missing", "ERROR")
            results[package] = False
    return results

def check_system_executables():
    """Check if system executables are available"""
    executables = {
        'tesseract': ['tesseract', '--version'],
        'poppler': ['pdftoppm', '-h']
    }
    
    results = {}
    for name, cmd in executables.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 or 'tesseract' in result.stderr.lower() or 'pdftoppm' in result.stderr.lower():
                print_status(f"System executable '{name}' - Available", "SUCCESS")
                results[name] = True
            else:
                print_status(f"System executable '{name}' - Not found", "ERROR")
                results[name] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_status(f"System executable '{name}' - Not found", "ERROR") 
            results[name] = False
    
    return results

def check_directory_structure():
    """Verify project directory structure"""
    required_dirs = ['src', 'data', 'imgs', 'vectorstore', 'logs', 'tests']
    base_path = Path('.')
    
    results = {}
    for directory in required_dirs:
        dir_path = base_path / directory
        if dir_path.exists():
            print_status(f"Directory '{directory}' - Exists", "SUCCESS")
            results[directory] = True
        else:
            print_status(f"Directory '{directory}' - Missing", "WARNING")
            # Create missing directory
            dir_path.mkdir(exist_ok=True)
            print_status(f"Created directory '{directory}'", "INFO")
            results[directory] = True
    
    return results

def check_environment_file():
    """Check if .env file exists and is readable"""
    env_file = Path('.env')
    if env_file.exists():
        print_status(".env file - Exists", "SUCCESS")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'CHUNK_SIZE' in content and 'EMBEDDING_MODEL' in content:
                    print_status(".env file - Configuration looks valid", "SUCCESS")
                    return True
                else:
                    print_status(".env file - Missing key configurations", "WARNING")
                    return False
        except Exception as e:
            print_status(f".env file - Error reading: {e}", "ERROR")
            return False
    else:
        print_status(".env file - Missing", "ERROR")
        return False

def main():
    """Run all verification checks"""
    print_status("Starting Multimodal RAG Setup Verification", "INFO")
    print("=" * 60)
    
    # Check Python version
    print_status("Checking Python Version...", "INFO")
    python_ok = check_python_version()
    print()
    
    # Check required packages
    print_status("Checking Python Packages...", "INFO")
    required_packages = [
        'unstructured', 'PIL', 'chromadb', 'sentence_transformers',
        'langchain', 'pandas', 'numpy', 'dotenv', 'tqdm', 'loguru'
    ]
    packages_ok = check_package_installation(required_packages)
    print()
    
    # Check system executables  
    print_status("Checking System Executables...", "INFO")
    executables_ok = check_system_executables()
    print()
    
    # Check directory structure
    print_status("Checking Directory Structure...", "INFO") 
    dirs_ok = check_directory_structure()
    print()
    
    # Check environment configuration
    print_status("Checking Environment Configuration...", "INFO")
    env_ok = check_environment_file()
    print()
    
    # Summary
    print("=" * 60)
    print_status("VERIFICATION SUMMARY", "INFO")
    print("=" * 60)
    
    if python_ok:
        print_status("✓ Python version compatible", "SUCCESS")
    else:
        print_status("✗ Python version incompatible", "ERROR")
    
    missing_packages = [pkg for pkg, status in packages_ok.items() if not status]
    if missing_packages:
        print_status(f"✗ Missing packages: {', '.join(missing_packages)}", "ERROR")
        print_status("Run: pip install -r requirements.txt", "INFO")
    else:
        print_status("✓ All required packages installed", "SUCCESS")
    
    missing_executables = [exe for exe, status in executables_ok.items() if not status]
    if missing_executables:
        print_status(f"✗ Missing executables: {', '.join(missing_executables)}", "ERROR")
        print_status("Install Tesseract OCR and Poppler utilities", "INFO")
    else:
        print_status("✓ All system executables available", "SUCCESS")
    
    if all(dirs_ok.values()):
        print_status("✓ Directory structure complete", "SUCCESS")
    else:
        print_status("⚠ Some directories were missing but have been created", "WARNING")
    
    if env_ok:
        print_status("✓ Environment configuration ready", "SUCCESS")
    else:
        print_status("✗ Environment configuration needs attention", "ERROR")
    
    # Final verdict
    all_critical_ok = (python_ok and 
                      all(packages_ok.values()) and 
                      all(executables_ok.values()) and
                      env_ok)
    
    if all_critical_ok:
        print_status("🎉 SETUP VERIFICATION PASSED - Ready to proceed!", "SUCCESS")
        return True
    else:
        print_status("❌ SETUP VERIFICATION FAILED - Please fix the issues above", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)