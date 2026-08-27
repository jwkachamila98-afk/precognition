"""Async WebSocket client (ws_client.py) for streaming compressed frames and receiving JSON telemetry."""

import asyncio
import logging
import time
from typing import Any, Optional
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from src.transport.protocol import FrameMessage, InferenceResponse, encode_image_to_base64

logger = logging.getLogger(__name__)


class WSStreamingClient:
    """
    Lightweight async WebSocket client.
    Connects to the backend server, transmits JPEG-compressed video frames along with intent metadata,
    and receives JSON telemetry containing MANO hand poses, depth maps, and 3D parsed scenes.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        server_url: Optional[str] = None,
        compression_quality: int = 80,
        timeout: float = 2.0
    ) -> None:
        if server_url:
            self.uri = server_url
            self.host = server_url
            self.port = 443 if server_url.startswith("wss://") else port
        else:
            self.host = host
            self.port = port
            self.uri = f"ws://{host}:{port}"
        self.compression_quality = compression_quality
        self.timeout = timeout
        self._ws: Optional[Any] = None
        self._is_connected = False

    async def connect(self) -> bool:
        """Establish async connection with the remote server (supports ws:// and wss://)."""
        try:
            try:
                from websockets.asyncio.client import connect as ws_connect
            except ImportError:
                from websockets import connect as ws_connect

            ssl_context = None
            if self.uri.startswith("wss://"):
                import ssl
                try:
                    import certifi
                    ssl_context = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    ssl_context = ssl._create_unverified_context()

            connect_kwargs = {
                "max_size": 10 * 1024 * 1024,  # 10 MB buffer
                "open_timeout": self.timeout
            }
            if ssl_context is not None:
                connect_kwargs["ssl"] = ssl_context

            self._ws = await ws_connect(self.uri, **connect_kwargs)
            self._is_connected = True
            logger.info(f"Connected to remote server at {self.uri}")
            return True
        except Exception as e:
            self._is_connected = False
            logger.warning(f"Failed to connect to {self.uri}: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        if hasattr(self._ws, "state"):
            return self._ws.state.name == "OPEN"
        if hasattr(self._ws, "closed"):
            return not self._ws.closed
        return self._is_connected

    async def send_frame(
        self,
        image: np.ndarray,
        frame_id: int,
        intent: str = "foresee me picking this remote control"
    ) -> Optional[InferenceResponse]:
        """
        Compress frame to JPEG, transmit over WebSocket with intent metadata, and await response.
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                return None

        h, w = image.shape[:2]
        now = time.time()

        try:
            b64_img = encode_image_to_base64(image, quality=self.compression_quality)
            frame_msg = FrameMessage(
                frame_id=frame_id,
                client_timestamp=now,
                image_base64=b64_img,
                width=w,
                height=h,
                intent=intent,
                compression="jpeg"
            )

            # Transmit frame request
            await self._ws.send(frame_msg.to_json())

            # Await inference response
            response_raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
            response = InferenceResponse.from_json(response_raw)
            return response

        except (ConnectionClosed, asyncio.TimeoutError) as e:
            logger.warning(f"WebSocket transport exception: {e}")
            self._is_connected = False
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error in ws_client: {e}")
            return None

    async def close(self) -> None:
        """Close WebSocket client connection gracefully."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._is_connected = False
        logger.info("WebSocket client closed.")


# Backwards compatibility alias
WebSocketStreamingClient = WSStreamingClient
