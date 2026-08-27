"""Compatibility module forwarding to ws_server.py."""

from src.transport.ws_server import WSInferenceServer, WebSocketInferenceServer

__all__ = ["WSInferenceServer", "WebSocketInferenceServer"]
