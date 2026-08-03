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

NOTE ON THE CROP UI: this used to use the third-party library
streamlit-drawable-canvas. That library turned out to have a frontend bug
that silently failed to render its background image once actually deployed
(confirmed via direct debugging that the image data itself was always
valid and correctly sized -- the failure was entirely inside that
library's own JS, with nothing left to fix from the Python side). Rather
than keep patching around an abandoned dependency, `corner_editor.py`
implements a small custom Streamlit component by hand, using the same
low-level protocol every component uses, with no third-party canvas
library involved.
"""

import io
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from corner_editor import corner_editor, image_to_jpeg_data_url
from flat_field import apply_flat_field
from painting_detect import find_corners_colormask, order_points, warp_to_rectangle

st.set_page_config(page_title="Painting Digitizer", layout="wide")

DISPLAY_MAX_DIM = 900
HANDLE_RADIUS = 6
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
        "skip_crop": False, # True if user chose to skip cropping entirely
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()


def clip_corners(corners, w, h):
    """Keep corners within the image bounds -- auto-detection (minAreaRect
    on a convex hull) can return points slightly outside the image edges,
    which puts a handle off-screen and undraggable."""
    corners = np.array(corners, dtype=float).copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners


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
            st.image(img_rgb, caption=name, use_container_width=True)
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

    scale = min(1.0, DISPLAY_MAX_DIM / max(h, w))
    disp_w, disp_h = int(w * scale), int(h * scale)
    disp_img = Image.fromarray(
        cv2.cvtColor(cv2.resize(img_full, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
    )
    data_url = image_to_jpeg_data_url(disp_img)

    corners_to_show = st.session_state.corners[name]
    disp_corners = (corners_to_show * scale).tolist()

    result = corner_editor(
        background_data_url=data_url,
        width=disp_w,
        height=disp_h,
        corners=disp_corners,
        handle_radius=HANDLE_RADIUS,
        edge_colors=EDGE_COLORS,
        key=f"corner_editor_{name}",
    )

    if result is not None and "corners" in result:
        new_disp_corners = np.array(result["corners"], dtype=float)
        new_corners = clip_corners(new_disp_corners / scale, w, h)
        if not np.allclose(new_corners, corners_to_show, atol=0.01):
            st.session_state.corners[name] = new_corners
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
        st.image(get_result_rgb(name), use_container_width=True)
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
