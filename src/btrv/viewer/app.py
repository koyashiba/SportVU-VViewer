from __future__ import annotations

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="btrv-viewer", description="Basketball Tracking & Research Viewer")
    p.add_argument("video", nargs="?", help="video file to open")
    p.add_argument("--run", help="run directory to load")
    p.add_argument("--source-id")
    p.add_argument("--port", type=int, default=8765, help="localhost adapter port")
    p.add_argument("--no-adapter", action="store_true")
    a = p.parse_args(argv)
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(sys.argv[:1])
    win = MainWindow(adapter_port=a.port, start_adapter=not a.no_adapter)
    win.show()
    if a.video:
        win.open_video(a.video, a.source_id)
    if a.run:
        win.load_run(a.run)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
