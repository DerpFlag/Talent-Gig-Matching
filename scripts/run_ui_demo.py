"""Launch the minimal Streamlit demo (API recommendations only)."""
import os
import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8502"]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
