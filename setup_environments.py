#!/usr/bin/env python3
"""
Environment Setup Script for Multi-World Trajectory Retrieval

Usage:
    python setup_environments.py --worlds alfworld
    python setup_environments.py --list

The data root defaults to the ``MATM_DATA_ROOT`` environment variable (or
``./environments`` if unset). See ``.env.example``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# GitHub repositories
ALFWORLD_REPO = "https://github.com/alfworld/alfworld.git"

# Environment base directory (override with the MATM_DATA_ROOT environment variable)
ENVIRONMENTS_DIR = os.environ.get(
    "MATM_DATA_ROOT", str(Path(__file__).parent.absolute() / "environments")
)

# AlfWorld configuration
# data/ = raw game JSON files (input)
# traj_logs/ = processed trajectories (output)
# indices/ = FAISS indices (output)
ALFWORLD_ENV_DIR = f"{ENVIRONMENTS_DIR}/alfworld"
ALFWORLD_DATA = f"{ALFWORLD_ENV_DIR}/data"
ALFWORLD_TRAJLOGS = f"{ALFWORLD_ENV_DIR}/traj_logs"
ALFWORLD_INDICES = f"{ALFWORLD_ENV_DIR}/indices"

# AlfWorld data download URLs (if available)
# Note: AlfWorld data is typically included in the package
ALFWORLD_DATA_URL = "https://github.com/alfworld/alfworld/raw/master/data.zip"

# ============================================================================


class EnvironmentSetup:
    """Base class for environment setup"""

    def __init__(self, root_dir: Path, force: bool = False):
        self.root_dir = root_dir
        # Use global ENVIRONMENTS_DIR instead of relative path
        self.env_dir = Path(ENVIRONMENTS_DIR) / self.name
        self.force = force

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def requirements_file(self) -> str:
        return f"requirements-{self.name}.txt"

    def create_directories(self):
        """Create directory structure for this environment"""
        dirs = [
            self.env_dir,
            self.env_dir / "goldpaths",
            self.env_dir / "data",
            self.env_dir / "configs",
            self.env_dir / "outputs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory structure for {self.name}")

    def install_dependencies(self):
        """Install Python dependencies"""
        req_file = self.root_dir / self.requirements_file
        if not req_file.exists():
            print(f"⚠ Requirements file {self.requirements_file} not found, skipping")
            return

        print(f"Installing dependencies from {self.requirements_file}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            )
            print(f"✓ Dependencies installed for {self.name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            raise

    def download_resources(self):
        """Download environment-specific resources (override in subclasses)"""
        pass

    def setup(self, download_only: bool = False):
        """Main setup method"""
        print(f"\n{'='*60}")
        print(f"Setting up {self.name.upper()}")
        print(f"{'='*60}\n")

        self.create_directories()

        if not download_only:
            self.install_dependencies()

        self.download_resources()

        # Create a marker file to indicate setup is complete
        marker_file = self.env_dir / ".setup_complete"
        with open(marker_file, "w") as f:
            json.dump(
                {"environment": self.name, "version": "1.0", "status": "complete"},
                f,
                indent=2,
            )

        print(f"✓ {self.name} setup complete!")

    def is_installed(self) -> bool:
        """Check if environment is already installed"""
        marker = self.env_dir / ".setup_complete"
        return marker.exists() and self.env_dir.exists()


class AlfWorldSetup(EnvironmentSetup):
    """Setup for AlfWorld environment"""

    @property
    def name(self) -> str:
        return "alfworld"

    def create_directories(self):
        """Create AlfWorld-specific directory structure"""
        # Create environment folder structure
        # data/ = raw game JSON files from alfworld-download (input)
        # traj_logs/ = processed trajectories (output from preprocessing)
        # indices/ = FAISS indices (output from preprocessing)
        self.env_dir.mkdir(parents=True, exist_ok=True)
        (self.env_dir / "data").mkdir(exist_ok=True)
        (self.env_dir / "traj_logs").mkdir(exist_ok=True)
        (self.env_dir / "indices").mkdir(exist_ok=True)

        print(f"✓ Created AlfWorld directory structure")

    def download_resources(self):
        """Download AlfWorld package and data"""
        print("Setting up AlfWorld environment...")

        env_data = self.env_dir / "data"

        # 1. Download AlfWorld package
        alfworld_pkg = self.env_dir / "alfworld"
        if alfworld_pkg.exists() and not self.force:
            print("✓ AlfWorld package already exists")
        else:
            print("Downloading AlfWorld package from GitHub...")
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    subprocess.check_call(
                        ["git", "clone", "--depth", "1", ALFWORLD_REPO, tmpdir],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    src_pkg = Path(tmpdir) / "alfworld"
                    if src_pkg.exists():
                        if alfworld_pkg.exists():
                            shutil.rmtree(alfworld_pkg)
                        shutil.copytree(src_pkg, alfworld_pkg)
                        print("✓ AlfWorld package downloaded")
                    else:
                        print("⚠ AlfWorld package not found in repo")

                    # Copy supporting files from repo
                    src_data = Path(tmpdir) / "data"
                    if src_data.exists():
                        print("Copying supporting files (alfred.pddl, alfred.twl2)...")
                        for item in src_data.iterdir():
                            if item.is_file():
                                dest = env_data / item.name
                                shutil.copy2(item, dest)
                        print("✓ Supporting files copied")

            except subprocess.CalledProcessError as e:
                print(f"⚠ Failed to clone AlfWorld repo: {e}")
            except Exception as e:
                print(f"⚠ Error downloading AlfWorld: {e}")

        # 2. Download AlfWorld TextWorld game data (json_2.1.1/) + seq2seq expert demos
        json_data = env_data / "json_2.1.1"
        if json_data.exists() and not self.force:
            print("✓ AlfWorld TextWorld game data already exists")
        else:
            print("Downloading AlfWorld TextWorld data + expert demonstrations...")
            print("(Includes: game files + pre-recorded TextWorld trajectories)")
            print("This may take a few minutes (~200MB with --extra)...")

            try:
                # Set ALFWORLD_DATA environment variable
                env = os.environ.copy()
                env["ALFWORLD_DATA"] = str(env_data)

                # Try to run alfworld-download WITH --extra to get seq2seq trajectories
                result = subprocess.run(
                    [
                        "alfworld-download",
                        "--extra",
                    ],  # --extra downloads pre-recorded expert demos
                    env=env,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print("✓ AlfWorld data downloaded successfully")
                    print("  ✓ TextWorld game files (json_2.1.1/)")
                    print(
                        "  ✓ Expert demonstrations (seq2seq data with correct actions)"
                    )
                else:
                    print("⚠ alfworld-download --extra command failed")
                    print(f"  Output: {result.stderr}")
                    raise Exception("Download failed")

            except FileNotFoundError:
                print("⚠ alfworld-download command not found")
                print("  Trying Python script method...")

                try:
                    # Try using Python API
                    result = subprocess.run(
                        [sys.executable, "-m", "alfworld.scripts.alfworld_download"],
                        env=env,
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        print("✓ AlfWorld game data downloaded")
                    else:
                        raise Exception("Python download failed")

                except Exception as e:
                    print(f"⚠ Could not download AlfWorld data automatically: {e}")
                    print("")
                    print("=" * 60)
                    print("MANUAL SETUP REQUIRED")
                    print("=" * 60)
                    print("Run these commands to download AlfWorld data:")
                    print(f"  export ALFWORLD_DATA={env_data}")
                    print("  alfworld-download --extra")
                    print("")
                    print("This downloads:")
                    print("  • TextWorld game files (json_2.1.1/)")
                    print("  • Expert demonstration trajectories (seq2seq data)")
                    print("")
                    print("Or manually download and place in:")
                    print(f"  {env_data}/")
                    print("=" * 60)
            except Exception as e:
                print(f"⚠ Error: {e}")

        # Verify TextWorld data and expert demos
        if json_data.exists():
            num_json = sum(1 for _ in json_data.rglob("*.json"))
            num_tw_games = sum(1 for _ in json_data.rglob("game.tw-pddl"))
            print(f"✓ AlfWorld data verified:")
            print(f"  • {num_json} JSON files")
            print(f"  • {num_tw_games} TextWorld game files")

            # Check for expert trajectories
            expert_files = list(json_data.rglob("expert_demo.txt")) + list(
                json_data.rglob("*_trajs.json")
            )
            if expert_files:
                print(f"  • {len(expert_files)} expert demonstration files found")
            else:
                print(f"  ⚠ No expert demonstration files found (did --extra work?)")
        else:
            print("⚠ AlfWorld data not found")

        print(f"✓ AlfWorld setup complete")
        print(f"  Package: {alfworld_pkg}")
        print(f"  Raw data: {env_data}")
        print(f"  Traj logs (output): {self.env_dir / 'traj_logs'}")
        print(f"  Indices (output): {self.env_dir / 'indices'}")
        print(
            f"\n  Note: Includes TextWorld expert demonstrations with correct action syntax"
        )
        print(f"  Example actions: 'go to desk 1', 'take alarmclock 2 from desk 1'")


# Registry of available environments
ENVIRONMENTS = {
    "alfworld": AlfWorldSetup,
}


def list_environments(root_dir: Path):
    """List all available and installed environments"""
    print(f"\n{'='*60}")
    print(f"Available Environments")
    print(f"{'='*60}\n")

    for env_name, env_class in ENVIRONMENTS.items():
        env_setup = env_class(root_dir)
        status = "✓ Installed" if env_setup.is_installed() else "✗ Not installed"
        print(f"  {env_name:15} {status}")
        if env_setup.is_installed():
            print(f"    Location: {env_setup.env_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup and manage environments for trajectory retrieval research",
    )

    parser.add_argument(
        "--worlds",
        nargs="+",
        choices=list(ENVIRONMENTS.keys()),
        help="Environments to setup (space-separated)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available environments and their status",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download resources, skip dependency installation",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of resources even if they exist",
    )

    args = parser.parse_args()

    root_dir = Path(__file__).parent.absolute()

    # Ensure the base environments directory exists
    Path(ENVIRONMENTS_DIR).mkdir(parents=True, exist_ok=True)

    # Handle list command
    if args.list:
        list_environments(root_dir)
        return

    # Require --worlds for setup
    if not args.worlds:
        print(
            "Please specify --worlds to setup, or use --list to see available environments"
        )
        parser.print_help()
        sys.exit(1)

    # Setup each specified environment
    for world in args.worlds:
        if world not in ENVIRONMENTS:
            print(f"✗ Unknown environment: {world}")
            continue

        env_setup = ENVIRONMENTS[world](root_dir, force=args.force)

        try:
            env_setup.setup(download_only=args.download_only)
        except Exception as e:
            print(f"✗ Failed to setup {world}: {e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Setup Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
