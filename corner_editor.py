"""
Python side of a small, hand-written Streamlit component for dragging the
4 corners of a quad over a background image. See corner_editor_frontend/index.html
for the JS half and the protocol it implements.

Built after streamlit-drawable-canvas (a third-party, unmaintained library)
turned out to have a frontend bug that silently failed to render its
background image once actually deployed, with the failure confirmed to be
entirely inside that library's own JS. This has no third-party canvas
dependency at all -- just the same low-level component protocol every
Streamlit component (including that one) is built on.

IMPORTANT: this uses declare_component(url=...) rather than path=...
Streamlit's local static-file serving for custom, locally-authored
components (as opposed to ones installed from PyPI) is a documented,
known-flaky area on some hosting platforms -- see
https://github.com/streamlit/streamlit/issues/9465, which describes this
exact symptom ("trouble loading the component... frontend assets") across
several different custom components on various deployment platforms.

An initial attempt pointed url= directly at raw.githubusercontent.com.
That turned out to be actively hostile to this use case -- checked
directly with `curl -I` against the real URL, which confirmed GitHub
serves raw files with three separate blockers stacked together:
  x-frame-options: deny                     (refuses ANY iframe embedding)
  content-security-policy: ...; sandbox     (blocks script execution even if loaded)
  content-type: text/plain                  (browser won't parse/run it as HTML at all)
A second attempt tried jsDelivr's GitHub-mirroring CDN, which fixed the
framing issue (no x-frame-options sent) but turned out to just relay
GitHub's own content-type metadata rather than inferring it fresh from
the .html extension -- so it was STILL served as text/plain, and with
x-content-type-options: nosniff also present, the browser won't guess
otherwise.

The actual fix: GitHub Pages, GitHub's real static-site hosting product
(as opposed to its deliberately-locked-down raw-content endpoint).
Verified directly via `curl -I` against the live Pages URL: correct
`content-type: text/html`, no framing-restriction headers, 200 status,
and a content-length exactly matching this file's real size.

If you fork/rename this repo, update RAW_HTML_URL below to match.
"""

import base64
import io

import streamlit.components.v1 as components

RAW_HTML_URL = "https://vlloyd1005.github.io/painting-digitizer/corner_editor_frontend/index.html"

_corner_editor_component = components.declare_component(
    "corner_editor",
    url=RAW_HTML_URL,
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
