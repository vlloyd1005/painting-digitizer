"""
Multi-stage painting digitization workflow:

  Stage 1 (upload):  upload raw session photos + a flat-field reference
                      photo (a clean patch of wall). Apply flat-field
                      correction to each raw photo.
  Stage 2 (review):  look at each corrected image, check off the ones
                      that look right. Any left unchecked can be
                      re-corrected with a different flat field before
                      continuing.
  Stage 3 (crop):    one image at a time. Auto-detected corners are shown
                      as a translucent blue quad with colored edges and
                      draggable corner handles -- all four are draggable
                      immediately, no need to select one first. Next/Back
                      to move between images.
  Stage 4 (export):  straighten + crop every approved image at full
                      resolution and download as a zip of high-quality JPEGs.

Run with: streamlit run app.py
"""

import base64
import io
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import streamlit_drawable_canvas as _sdc
from streamlit_drawable_canvas import st_canvas

from flat_field import apply_flat_field
from painting_detect import find_corners_colormask, order_points, warp_to_rectangle

# streamlit-drawable-canvas renders inside a sandboxed iframe and asks
# Streamlit to serve the background image at a server-relative URL (e.g.
# "/media/abc123.png"). That resolves fine on localhost, where the iframe
# happens to share the same origin as the app -- but once actually
# deployed, that relative path can resolve against the wrong origin/base,
# 404, and the canvas shows solid black because the browser never loads
# the image. Fix: make the "URL" a self-contained base64 data URI instead,
# which has no path/origin to resolve.
#
# IMPORTANT: earlier attempts patched streamlit.elements.image.image_to_url
# directly (temporarily, via try/finally around the st_canvas() call). That
# function is also what Streamlit's own st.image() widget uses internally,
# and the temporary-swap approach turned out not to reliably un-patch itself
# on Streamlit Cloud (likely because Streamlit's own rerun mechanism can
# interrupt script execution in a way that doesn't behave like an ordinary
# Python exception, so a `finally` block isn't guaranteed to run before the
# next script run starts) -- it kept leaking into st.image() calls elsewhere.
#
# This version is leak-proof by construction: streamlit_drawable_canvas's
# own module has its OWN name "st_image" bound to the real
# streamlit.elements.image module. We replace THAT one name, inside
# streamlit_drawable_canvas's own namespace only. Streamlit's own st.image()
# widget looks up image_to_url through its own module's namespace directly
# and never goes through streamlit_drawable_canvas.st_image at all, so it's
# completely unaffected -- there is nothing to restore, and nothing to leak.
def _image_to_data_url(image, width=None, clamp=False, channels="RGB", output_format="PNG", image_id=""):
    fmt = (output_format or "PNG").upper()
    if fmt not in ("PNG", "JPEG"):
        fmt = "PNG"
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

class _FakeStImageModule:
    image_to_url = staticmethod(_image_to_data_url)

_sdc.st_image = _FakeStImageModule()

st.set_page_config(page_title="Painting Digitizer", layout="wide")


DISPLAY_MAX_DIM = 900
HANDLE_RADIUS = 3
EDGE_LINE_WIDTH = 1
EDGE_COLORS = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261"]  # top, right, bottom, left

# ---------------------------------------------------------------- helpers

def load_bgr(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def init_state():
    defaults = {
        "stage": "upload",
        "images": {},       # name -> corrected BGR image (full res)
        "approved": {},     # name -> bool
        "crop_order": [],   # list of names going into crop stage
        "crop_index": 0,
        "corners": {},      # name -> 4x2 array (full-res coords)
        "canvas_rev": {},   # name -> int, bumped to force-remount canvas
        "skip_crop": False, # True if user chose to skip cropping entirely
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()


# ---------------------------------------------------------------- stage 1

def stage_upload():
    st.title("Step 1 — Flat-field correction")
    st.write(
        "Upload the raw photos from this session, plus one **flat field** photo: "
        "a shot of a clean patch of the same wall (no nails, no shadows, no "
        "paintings) taken under the same lighting. This cancels out uneven "
        "lighting and sensor artifacts the same way it does for astro imaging."
    )
    if st.session_state.images:
        st.caption(f"({len(st.session_state.images)} image(s) already corrected so far -- uploading more here adds to that batch.)")

    raws = st.file_uploader(
        "Raw session photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    flat = st.file_uploader("Flat field photo (clean wall patch)", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply flat-field correction", type="primary", disabled=not (raws and flat)):
            flat_bgr = load_bgr(flat)
            for raw in raws:
                corrected = apply_flat_field(load_bgr(raw), flat_bgr)
                st.session_state.images[raw.name] = corrected
                st.session_state.approved[raw.name] = False
            st.session_state.stage = "review"
            st.rerun()
    with col2:
        if st.button("Skip this step (just crop, no correction)", disabled=not raws):
            for raw in raws:
                st.session_state.images[raw.name] = load_bgr(raw)
                st.session_state.approved[raw.name] = False
            st.session_state.stage = "review"
            st.rerun()


# ---------------------------------------------------------------- stage 2

def stage_review():
    st.title("Step 2 — Check the corrected images")
    st.write("Check off the images that look right. Anything left unchecked can be redone below with a different flat field.")

    names = list(st.session_state.images.keys())
    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i % 3]:
            img_rgb = cv2.cvtColor(st.session_state.images[name], cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=name, use_column_width=True)
            st.session_state.approved[name] = st.checkbox(
                "Looks correct", value=st.session_state.approved[name], key=f"chk_{name}"
            )

    st.divider()
    with st.expander("Some images still don't look right? Re-correct with a different flat field"):
        redo_raws = st.file_uploader(
            "Raw photos to redo", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="redo_raws"
        )
        redo_flat = st.file_uploader("New flat field photo", type=["jpg", "jpeg", "png"], key="redo_flat")
        if st.button("Re-correct these images", disabled=not (redo_raws and redo_flat)):
            flat_bgr = load_bgr(redo_flat)
            for raw in redo_raws:
                corrected = apply_flat_field(load_bgr(raw), flat_bgr)
                st.session_state.images[raw.name] = corrected
                st.session_state.approved[raw.name] = False
            st.rerun()

    st.divider()
    n_approved = sum(st.session_state.approved.values())
    st.write(f"**{n_approved} of {len(names)}** image(s) checked off.")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to upload"):
            st.session_state.stage = "upload"
            st.rerun()
    with col2:
        if st.button("Continue to Crop →", type="primary", disabled=n_approved == 0):
            st.session_state.crop_order = [n for n in names if st.session_state.approved[n]]
            st.session_state.crop_index = 0
            st.session_state.skip_crop = False
            st.session_state.stage = "crop"
            st.rerun()
    with col3:
        if st.button("Skip this step (download corrected images as-is)", disabled=n_approved == 0):
            st.session_state.crop_order = [n for n in names if st.session_state.approved[n]]
            st.session_state.skip_crop = True
            st.session_state.stage = "export"
            st.rerun()


# ---------------------------------------------------------------- stage 3

def build_canvas_objects(disp_corners):
    """4 draggable circles + translucent fill + 4 colored edge lines."""
    objects = [
        {
            "type": "polygon",
            "left": 0, "top": 0,
            "points": [{"x": float(p[0]), "y": float(p[1])} for p in disp_corners],
            "fill": "rgba(0, 120, 255, 0.25)",
            "stroke": "rgba(0,0,0,0)",
            "selectable": False,
            "evented": False,
        }
    ]
    n = len(disp_corners)
    for i in range(n):
        p1, p2 = disp_corners[i], disp_corners[(i + 1) % n]
        objects.append(
            {
                "type": "line",
                "x1": float(p1[0]), "y1": float(p1[1]),
                "x2": float(p2[0]), "y2": float(p2[1]),
                "stroke": EDGE_COLORS[i % len(EDGE_COLORS)],
                "strokeWidth": EDGE_LINE_WIDTH,
                "selectable": False,
                "evented": False,
            }
        )
    for p in disp_corners:
        objects.append(
            {
                "type": "circle",
                "left": float(p[0]) - HANDLE_RADIUS,
                "top": float(p[1]) - HANDLE_RADIUS,
                "radius": HANDLE_RADIUS,
                "fill": "rgba(0, 200, 0, 0.9)",
                "stroke": "white",
                "strokeWidth": 1,
                "hasControls": False,
            }
        )
    return objects


def clip_corners(corners, w, h):
    """Keep corners within the image bounds -- auto-detection (minAreaRect
    on a convex hull) can return points slightly outside the image edges,
    which puts a handle off-screen and undraggable."""
    corners = np.array(corners, dtype=float).copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners


def stage_crop():
    order = st.session_state.crop_order
    idx = st.session_state.crop_index
    name = order[idx]
    img_full = st.session_state.images[name]
    h, w = img_full.shape[:2]

    st.title(f"Step 3 — Adjust crop  ({idx + 1} of {len(order)})")
    st.caption(f"**{name}** — drag any corner to adjust the crop.")

    if name not in st.session_state.corners:
        detected = find_corners_colormask(img_full)
        if detected is None:
            m = 0.05
            detected = order_points(
                np.array([[w*m, h*m], [w*(1-m), h*m], [w*(1-m), h*(1-m)], [w*m, h*(1-m)]])
            )
        st.session_state.corners[name] = clip_corners(detected, w, h)
        st.session_state.canvas_rev[name] = 0

    scale = min(1.0, DISPLAY_MAX_DIM / max(h, w))
    disp_w, disp_h = int(w * scale), int(h * scale)
    disp_img = Image.fromarray(
        cv2.cvtColor(cv2.resize(img_full, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
    )

    # Passing initial_drawing=None does NOT mean "leave the canvas alone" --
    # per the library's own docs, None *empties* the canvas. So we must
    # always pass a real drawing. The original jitter came from a different
    # cause: floating-point drift between what we send and what the browser
    # echoes back, which never exactly converges. Rounding coordinates to
    # whole pixels on both the write and read side makes the round-trip
    # exact, so it settles instead of oscillating.
    corners_to_show = np.round(st.session_state.corners[name]).astype(float)
    st.session_state.corners[name] = corners_to_show
    disp_corners = np.round(corners_to_show * scale)

    canvas_result = st_canvas(
        background_image=disp_img,
        height=disp_h,
        width=disp_w,
        drawing_mode="transform",
        initial_drawing={"version": "4.4.0", "objects": build_canvas_objects(disp_corners)},
        key=f"canvas_{name}_{st.session_state.canvas_rev[name]}",
        display_toolbar=False,
    )

    if canvas_result.json_data is not None:
        circles = [o for o in canvas_result.json_data["objects"] if o["type"] == "circle"]
        if len(circles) == 4:
            returned_positions = np.array(
                [[c["left"] + c["radius"], c["top"] + c["radius"]] for c in circles]
            )
            # IMPORTANT: don't assume circles[i] is still "slot i". In
            # "transform" drawing mode, fabric.js brings the object you just
            # dragged to the front of its internal stack, which reorders the
            # serialized objects array. Matching by array index after that
            # silently swaps which stored corner gets which new position --
            # harmless-looking after one drag, visibly broken (oscillating)
            # after a second. Match each known slot to its NEAREST returned
            # circle instead, which is robust to that reordering.
            prev_disp = corners_to_show * scale
            used = set()
            new_disp_corners = np.zeros_like(prev_disp)
            for i, prev_pt in enumerate(prev_disp):
                dists = np.linalg.norm(returned_positions - prev_pt, axis=1)
                for j in used:
                    dists[j] = np.inf
                best = int(np.argmin(dists))
                new_disp_corners[i] = returned_positions[best]
                used.add(best)
            new_disp_corners = np.round(new_disp_corners)
            new_corners = np.round(new_disp_corners / scale)
            new_corners = clip_corners(new_corners, w, h)
            if not np.allclose(new_corners, corners_to_show, atol=0.01):
                st.session_state.corners[name] = new_corners
                # Force a full remount (new widget key) rather than updating
                # initial_drawing on the same persistent component instance.
                # Repeatedly reloading content into a long-lived instance can
                # leave stale/duplicate objects behind instead of cleanly
                # replacing them -- a fresh mount avoids that entirely.
                st.session_state.canvas_rev[name] += 1
                st.rerun()

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if idx > 0:
            if st.button("← Back"):
                st.session_state.crop_index -= 1
                st.rerun()
        else:
            if st.button("← Back to review"):
                st.session_state.stage = "review"
                st.rerun()
    with col2:
        label = "Finish →" if idx == len(order) - 1 else "Next →"
        if st.button(label, type="primary"):
            if idx == len(order) - 1:
                st.session_state.stage = "export"
            else:
                st.session_state.crop_index += 1
            st.rerun()
    with col3:
        if st.button("Reset this image's corners to auto-detected guess"):
            detected = find_corners_colormask(img_full)
            if detected is not None:
                st.session_state.corners[name] = clip_corners(detected, w, h)
                st.session_state.canvas_rev[name] += 1
                st.rerun()


# ---------------------------------------------------------------- stage 4

def stage_export():
    st.title("Step 4 — Preview & Download")
    order = st.session_state.crop_order
    skip_crop = st.session_state.skip_crop

    def get_result_rgb(name):
        img_full = st.session_state.images[name]
        if skip_crop:
            return cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        corners = st.session_state.corners[name]
        return cv2.cvtColor(warp_to_rectangle(img_full, corners), cv2.COLOR_BGR2RGB)

    if skip_crop:
        st.write("Cropping was skipped -- these are the corrected images as-is.")
    else:
        st.write("Check each result at full size. If one needs adjustment, re-crop it -- it'll pick up right where you left off.")

    for name in order:
        st.subheader(name)
        st.image(get_result_rgb(name), use_column_width=True)
        if not skip_crop:
            if st.button("Re-crop this image", key=f"recrop_{name}"):
                st.session_state.crop_index = order.index(name)
                st.session_state.stage = "crop"
                st.rerun()
        st.divider()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in order:
            buf = io.BytesIO()
            Image.fromarray(get_result_rgb(name)).save(buf, format="JPEG", quality=95)
            suffix = "_corrected.jpg" if skip_crop else "_cropped.jpg"
            out_name = name.rsplit(".", 1)[0] + suffix
            zf.writestr(out_name, buf.getvalue())

    st.success(f"{len(order)} image(s) ready.")
    st.download_button(
        "Download all as ZIP",
        data=zip_buf.getvalue(),
        file_name="cropped_paintings.zip",
        mime="application/zip",
        type="primary",
    )

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to crop"):
            st.session_state.crop_index = len(order) - 1
            st.session_state.stage = "crop"
            st.rerun()
    with col2:
        if st.button("Start over"):
            reset_all()
            st.rerun()


# ---------------------------------------------------------------- main

init_state()

stages = {
    "upload": stage_upload,
    "review": stage_review,
    "crop": stage_crop,
    "export": stage_export,
}
stages[st.session_state.stage]()
