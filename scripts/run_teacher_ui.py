"""Launch the step-by-step teacher UI (Learn & run, guide sections, Try API)."""
import os
import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/teach_app.py", "--server.port", "8503"]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
