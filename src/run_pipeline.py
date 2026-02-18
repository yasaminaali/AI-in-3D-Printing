#!/usr/bin/env python3
"""
Entry point script for running the SA dataset generation pipeline.
Auto-creates virtual environment and installs dependencies if needed.

Usage:
    python run_pipeline.py <machine_id> [--workers N] [--retry-failed] [--config-dir DIR]

Examples:
    python run_pipeline.py yasamin --workers 8
    python run_pipeline.py istiaq
    python run_pipeline.py kazi --retry-failed
"""

import argparse
import os
import sys
import subprocess
import platform


def get_venv_path(script_dir):
    """Get virtual environment path."""
    return os.path.join(script_dir, ".venv")


def get_python_executable(venv_path):
    """Get Python executable path in venv."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def get_pip_executable(venv_path):
    """Get pip executable path in venv."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_path, "bin", "pip")


def venv_exists(venv_path):
    """Check if virtual environment exists."""
    python_exe = get_python_executable(venv_path)
    return os.path.exists(python_exe)


def create_venv(venv_path):
    """Create virtual environment."""
    print(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
        print("✓ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False


def install_dependencies(venv_path, script_dir):
    """Install dependencies from requirements.txt."""
    pip_exe = get_pip_executable(venv_path)
    requirements_file = os.path.join(script_dir, "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print(f"✗ requirements.txt not found at {requirements_file}")
        return False
    
    print("Installing dependencies from requirements.txt...")
    try:
        # Upgrade pip first
        subprocess.check_call([pip_exe, "install", "--upgrade", "pip"])
        # Install requirements
        subprocess.check_call([pip_exe, "install", "-r", requirements_file])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def check_and_setup_venv(script_dir):
    """Check venv exists, create if needed, install dependencies."""
    venv_path = get_venv_path(script_dir)
    
    # Check if we're already in the venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Already in a virtual environment, check if it's the right one
        if sys.prefix == venv_path:
            return True
    
    # Check if venv exists
    if not venv_exists(venv_path):
        print("Virtual environment not found. Setting up...")
        if not create_venv(venv_path):
            return False
        if not install_dependencies(venv_path, script_dir):
            return False
    else:
        # Venv exists, check if we need to install/update dependencies
        python_exe = get_python_executable(venv_path)
        try:
            # Try to import rich to check if dependencies are installed
            result = subprocess.run(
                [python_exe, "-c", "import rich, yaml, tqdm"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("Dependencies missing or outdated. Installing...")
                if not install_dependencies(venv_path, script_dir):
                    return False
        except Exception:
            print("Checking dependencies failed. Reinstalling...")
            if not install_dependencies(venv_path, script_dir):
                return False
    
    return True


def run_in_venv(venv_path, script_dir, args):
    """Re-run the script in the virtual environment."""
    python_exe = get_python_executable(venv_path)
    
    # Build command line arguments
    cmd = [python_exe, __file__] + sys.argv[1:]
    
    print(f"Running in virtual environment...")
    try:
        # Execute the script in the venv
        result = subprocess.run(cmd, cwd=script_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error running in virtual environment: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run SA dataset generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py yasamin              # Run Yasamin's tasks with default workers
  python run_pipeline.py istiaq --workers 4   # Run Istiaq's tasks with 4 workers
  python run_pipeline.py kazi --retry-failed  # Retry failed tasks for Kazi
        """,
    )

    parser.add_argument(
        "machine_id",
        type=str,
        help="Machine identifier (e.g., yasamin, istiaq, kazi)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: from config or CPU count)",
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML config files (default: config)",
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed tasks",
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Seconds between progress logs (default: 60)",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status and exit without running",
    )

    parser.add_argument(
        "--skip-venv-check",
        action="store_true",
        help="Skip virtual environment check (use current Python)",
    )

    args = parser.parse_args()

    # Resolve script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = get_venv_path(script_dir)
    
    # Check if we need to setup/run in venv
    if not args.skip_venv_check:
        # Check if we're already in the correct venv
        current_python = sys.executable
        expected_python = get_python_executable(venv_path)
        
        if current_python != expected_python and venv_exists(venv_path):
            # We're not in the venv, run in it
            run_in_venv(venv_path, script_dir, args)
            return
        elif not venv_exists(venv_path):
            # Venv doesn't exist, create it
            if not check_and_setup_venv(script_dir):
                print("✗ Failed to setup virtual environment")
                sys.exit(1)
            # Now run in the newly created venv
            run_in_venv(venv_path, script_dir, args)
            return
        elif current_python == expected_python:
            # We're in the venv, check if dependencies are installed using subprocess
            python_exe = get_python_executable(venv_path)
            deps_installed = False
            try:
                result = subprocess.run(
                    [python_exe, "-c", "import rich, yaml, tqdm, numpy, matplotlib"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                deps_installed = (result.returncode == 0)
            except Exception:
                deps_installed = False
            
            if not deps_installed:
                print("Dependencies not installed in virtual environment. Installing...")
                if not install_dependencies(venv_path, script_dir):
                    print("✗ Failed to install dependencies")
                    sys.exit(1)
                print("✓ Dependencies installed successfully")
                # Continue execution - imports will work now
    
    # We're in the correct venv or skipping venv check, run the actual pipeline
    # Add current directory to path for imports
    sys.path.insert(0, script_dir)
    
    # Import after confirming we're in venv with dependencies
    try:
        from pipeline.runner import run_pipeline, ParallelRunner
    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        print("Please ensure dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    # Resolve config directory relative to script location
    config_dir = os.path.join(script_dir, args.config_dir)

    if not os.path.isdir(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    # Check if machine config exists
    machine_config_path = os.path.join(config_dir, f"{args.machine_id}.yaml")
    if not os.path.exists(machine_config_path):
        print(f"Error: Machine config not found: {machine_config_path}")
        available = [f.replace(".yaml", "") for f in os.listdir(config_dir) if f.endswith(".yaml") and f != "global_config.yaml"]
        print(f"Available machines: {', '.join(available)}")
        sys.exit(1)

    if args.status:
        # Show status only
        runner = ParallelRunner(
            config_dir=config_dir,
            machine_id=args.machine_id,
            num_workers=args.workers,
        )
        status = runner.get_status()

        print(f"\n=== Status for {args.machine_id} ===")
        print(f"Total tasks: {status['total_tasks']}")
        print(f"Completed: {status['progress']['completed']}")
        print(f"Failed: {status['progress']['failed']}")
        print(f"Remaining: {status['progress']['remaining']}")
        print(f"Progress: {status['progress']['percent']:.1f}%")
        return

    # Run the pipeline
    print(f"\n=== SA Dataset Generation Pipeline ===")
    print(f"Machine: {args.machine_id}")
    print(f"Config dir: {config_dir}")
    print()

    try:
        stats = run_pipeline(
            config_dir=config_dir,
            machine_id=args.machine_id,
            num_workers=args.workers,
            retry_failed=args.retry_failed,
            progress_interval=args.progress_interval,
        )

        # Exit with error code if there were failures
        if stats.get("failed_count", 0) > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
