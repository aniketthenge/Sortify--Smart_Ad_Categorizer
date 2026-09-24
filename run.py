"""Start the web app.

Local use (default):   python run.py                -> http://127.0.0.1:5000 on this PC only
Serve customers:        set HOST=0.0.0.0, ADMIN_PASSWORD=<strong password>, then python run.py
Other settings:         PORT (5000), REFRESH_HOURS (6; 0 = off), NO_BROWSER=1
"""
import os
import sys
import webbrowser
from threading import Timer

try:
    from app.web import create_app
except ImportError as exc:
    if "Application Control policy" in str(exc):
        sys.exit(
            "\n  Windows blocked a Python component that this app needs:\n    " + str(exc) + "\n\n"
            "  This comes from Windows Smart App Control / company security policy, not from the app.\n"
            "  Options: run setup.bat again (freshly downloaded files are sometimes allowed on a second try),\n"
            "  ask IT to allow Python packages in this folder, or install Python 3.12 and re-run setup.bat.\n")
    raise


def open_in_chrome(url: str):
    """Open the site in Google Chrome; fall back to the default browser only if Chrome isn't installed."""
    import shutil
    import subprocess
    candidates = [shutil.which("chrome"), shutil.which("google-chrome"),
                  os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                  os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                  os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for exe in candidates:
        if exe and os.path.exists(exe):
            subprocess.Popen([exe, url])
            return
    print("  Google Chrome not found - opening your default browser instead.")
    webbrowser.open(url)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    public = host not in ("127.0.0.1", "localhost", "::1")
    if public and not os.environ.get("ADMIN_PASSWORD"):
        print("\n  NOTE: HOST is public but ADMIN_PASSWORD is not set, so the staff page is switched off.\n"
              "        Set ADMIN_PASSWORD to use /admin from other computers.\n")
    app = create_app()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    if os.environ.get("NO_BROWSER") != "1" and not public:
        Timer(1.5, lambda: open_in_chrome(url)).start()
    print(f"\n  Ad Categorizer running at {url}   (Ctrl+C to stop)\n")
    try:
        from waitress import serve  # production-grade server
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port, debug=False, threaded=True)
