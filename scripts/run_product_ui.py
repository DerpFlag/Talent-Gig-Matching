"""Launch the company-facing Streamlit app (same as run_ui.py)."""
import os
import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/product_app.py", "--server.port", "8501"]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
