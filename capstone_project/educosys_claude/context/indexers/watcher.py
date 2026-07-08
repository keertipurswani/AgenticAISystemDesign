from __future__ import annotations

import platform
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from educosys_claude.context.indexers.code_parser import ALL_EXTENSIONS
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# How the watcher works
# ------------------------------------------------------------------
# A watchdog Observer runs in a background daemon thread watching the
# project directory recursively. When a file changes, the OS fires an
# event → _CodebaseEventHandler receives it → debounces it (editors
# emit multiple events per save) → calls index_single_file() or
# remove_file_from_index() from semantic_chroma.py.
#
# This keeps ChromaDB up-to-date in real time so /ask queries reflect
# changes made by /plan tasks or manual edits without needing a restart.
# ------------------------------------------------------------------

# Editors often fire several events for a single save (write + chmod +
# rename swap). Wait this long after the last event before acting —
# any new event for the same file resets the timer.
_DEBOUNCE_SECONDS = 1.5


def _get_observer() -> Observer:
    """
    Return the best available watchdog observer for the current OS.

    macOS  → FSEvents   (native, event-driven, instant)
    Linux  → inotify    (native, event-driven, instant)
    Windows → PollingObserver — ReadDirectoryChangesW has edge cases
              under heavy load and on network drives, so we poll every
              2 seconds instead. Slightly delayed but reliable.
    """
    if platform.system() == "Windows":
        return PollingObserver(timeout=2)
    return Observer()


class _CodebaseEventHandler(FileSystemEventHandler):
    """
    Translates raw watchdog filesystem events into debounced indexer calls.

    Only reacts to files with extensions the indexer understands (ALL_EXTENSIONS
    from code_parser). Directory events and unrecognised extensions are ignored.

    Debounce pattern:
      Each file gets its own threading.Timer. If a new event arrives for the
      same file before the timer fires, the old timer is cancelled and a new
      one starts. The indexer is only called once the file has been stable for
      _DEBOUNCE_SECONDS.
    """

    def __init__(self, on_change=None) -> None:
        # {filepath: threading.Timer} — one pending debounced call per file.
        # Timers are daemon threads so they don't block process exit.
        self._timers: dict[str, threading.Timer] = {}
        self._on_change = on_change

    def _is_indexable(self, path: str) -> bool:
        """True if the file extension is one the indexer knows how to parse."""
        return Path(path).suffix.lower() in ALL_EXTENSIONS

    def _schedule(self, action: str, filepath: str) -> None:
        """
        Schedule a debounced indexer call for filepath.
        Cancels any already-pending call for the same file first.
        action is either "upsert" (create/modify) or "delete".
        """
        existing = self._timers.pop(filepath, None)
        if existing:
            existing.cancel()

        timer = threading.Timer(_DEBOUNCE_SECONDS, self._run, args=(action, filepath))
        timer.daemon = True
        timer.start()
        self._timers[filepath] = timer

    def _run(self, action: str, filepath: str) -> None:
        """
        Actual indexer call, executed after the debounce window expires.

        Imported inside the method to avoid a circular import at module load
        time (watcher ← semantic_chroma ← watcher would form a cycle).
        """
        self._timers.pop(filepath, None)
        from educosys_claude.context.indexers.semantic_chroma import (
            index_single_file,
            remove_file_from_index,
        )
        if action == "delete":
            remove_file_from_index(filepath)
        else:
            index_single_file(filepath)
        if self._on_change is not None:
            self._on_change()

    # ── watchdog event hooks ──────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("upsert", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("upsert", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_indexable(event.src_path):
            self._schedule("delete", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        # A rename/move = delete the old path + upsert the new path.
        # Both paths are checked independently since either could be non-indexable.
        if not event.is_directory:
            if self._is_indexable(event.src_path):
                self._schedule("delete", event.src_path)
            if self._is_indexable(event.dest_path):
                self._schedule("upsert", event.dest_path)


def start_watcher(repo_path: str, on_change=None) -> Observer:
    """
    Start a filesystem observer on repo_path in a background daemon thread.

    The observer watches recursively — all subdirectories are covered.
    Daemon=True means the thread won't prevent the process from exiting
    if the user hits Ctrl+C without going through /exit.

    on_change, if given, is called (with no args) after every debounced
    index update - used to invalidate the semantic cache when the
    underlying codebase changes.

    Returns the Observer so the caller can call stop_watcher() on shutdown.
    """
    handler  = _CodebaseEventHandler(on_change=on_change)
    observer = _get_observer()
    observer.schedule(handler, repo_path, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info(f"Watchdog started on {repo_path} (backend: {type(observer).__name__})")
    return observer


def stop_watcher(observer: Observer) -> None:
    """
    Cleanly stop the observer and wait for its thread to finish.
    Called in the finally block of _run_async() in main.py so it
    always runs regardless of how the REPL exits.
    """
    observer.stop()
    observer.join()
    logger.info("Watchdog stopped")