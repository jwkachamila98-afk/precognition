"""Make the default SSL context able to verify certificates.

A Python installed from python.org on macOS ships without a usable CA bundle
unless "Install Certificates.command" was run, so *anything* going through the
default ``ssl`` context fails with CERTIFICATE_VERIFY_FAILED while curl and git
work fine. This project already works around it in five places by building a
private ``certifi`` context (ws_client, gemini_vision_grounder, speech_to_text,
text_to_speech, mediapipe_tasks_hand_tracker) - but that only helps code we
own. Third-party downloaders take no such argument:

  * ``torch.hub.load`` (MiDaS depth) fails with a misleading "It looks like
    there is no internet connection" - the request never got past TLS;
  * ``ultralytics`` model fetches fail the same way.

Setting SSL_CERT_FILE/REQUESTS_CA_BUNDLE fixes the *default* context for the
whole process, so those libraries work without patching them.
"""

from __future__ import annotations

import logging
import os
import ssl

logger = logging.getLogger(__name__)

_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def ensure_ca_bundle() -> bool:
    """Point OpenSSL at certifi's bundle if the default context can't verify.

    Returns True if this call installed the bundle. Safe to call repeatedly, and
    never overrides a bundle the environment already specifies. Call it BEFORE
    importing anything that downloads on import.
    """
    if any(os.environ.get(v) for v in _VARS):
        return False

    try:
        ssl.create_default_context().load_default_certs()
        probe = ssl.create_default_context()
        if probe.cert_store_stats().get("x509_ca", 0) > 0:
            return False              # the interpreter already has a usable store
    except Exception:
        pass                          # fall through and install certifi's bundle

    try:
        import certifi
    except ImportError:
        logger.warning("certifi is not installed; HTTPS downloads may fail "
                       "certificate verification.")
        return False

    bundle = certifi.where()
    for var in _VARS:
        os.environ.setdefault(var, bundle)
    logger.info(f"SSL: no system CA bundle found; using certifi ({bundle}).")
    return True
