import subprocess
import sys


def main():
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
