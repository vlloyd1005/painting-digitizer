"""
Python side of a small, hand-written Streamlit component for dragging the
4 corners of a quad over a background image. See corner_editor/frontend/index.html
for the JS half and the protocol it implements.

Built after streamlit-drawable-canvas (a third-party, unmaintained library)
turned out to have a frontend bug that silently failed to render its
background image once actually deployed, with the failure confirmed to be
entirely inside that library's own JS. This has no third-party canvas
dependency at all -- just the same low-level component protocol every
Streamlit component (including that one) is built on.
"""

import base64
import io
import os

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corner_editor", "frontend")

_corner_editor_component = components.declare_component(
    "corner_editor",
    path=_FRONTEND_DIR,
)


def corner_editor(background_data_url, width, height, corners, handle_radius=6,
                   edge_colors=None, key=None):
    """
    background_data_url: a "data:image/...;base64,..." string
    width, height: display size in pixels
    corners: list of 4 [x, y] pairs in the same pixel space as width/height
    Returns: None (no drag yet) or {"corners": [[x,y], [x,y], [x,y], [x,y]]}
             reflecting the latest drag-and-release.
    """
    if edge_colors is None:
        edge_colors = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261"]
    return _corner_editor_component(
        background_data_url=background_data_url,
        width=width,
        height=height,
        corners=corners,
        handle_radius=handle_radius,
        edge_colors=edge_colors,
        key=key,
        default=None,
    )


def image_to_jpeg_data_url(pil_image, quality=85):
    """Encode a PIL Image as a JPEG data URI (small, no path/origin to resolve)."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
