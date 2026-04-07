#!/usr/bin/env python3
"""Quick wrapper to run main experiment with a single seed."""
import subprocess, sys, os
args = sys.argv[1:]
cmd = [sys.executable, "scripts/full_experiment.py"] + args
subprocess.run(cmd)
