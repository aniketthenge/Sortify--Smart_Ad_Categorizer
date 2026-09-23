"""Start the web app:  python run.py   ->  http://127.0.0.1:5000"""
import os
import webbrowser
from threading import Timer

from app.web import create_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app = create_app()
    if os.environ.get("NO_BROWSER") != "1":
        Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  Ad Categorizer running at http://127.0.0.1:{port}   (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
