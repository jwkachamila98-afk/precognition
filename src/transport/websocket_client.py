"""Compatibility module forwarding to ws_client.py."""

from src.transport.ws_client import WSStreamingClient, WebSocketStreamingClient

__all__ = ["WSStreamingClient", "WebSocketStreamingClient"]
