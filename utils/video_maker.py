import numpy as np
import cv2
import imageio
import torch
import matplotlib.cm as cm
import warnings
import math
def _to_uint8_rgb(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 3:  # C,H,W
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    if x.shape[2] == 4:
        x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)
    return x

def _apply_colormap_gray(gray, cmap_name='jet'):  # Default to 'jet' for warm colors for near objects
    arr = np.asarray(gray)
    if np.issubdtype(arr.dtype, np.floating):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            norm = np.zeros_like(arr, dtype=np.float32)
        else:
            norm = (arr - mn) / (mx - mn)
        rgba = cm.get_cmap(cmap_name)(norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    else:
        norm = arr.astype(np.float32) / 255.0
        rgba = cm.get_cmap(cmap_name)(norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb
def visualize_depth(x, acc=None, lo=4.0, hi=120.0, depth_curve_fn=lambda x: -np.log(x + 1e-6), depth_bias=0.0):
    depth = np.asarray(x).astype(np.float32)
    # ensure lo <= hi
    lo_f = float(lo)
    hi_f = float(hi)
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    min_allowed = lo_f + 0.01
    if min_allowed > hi_f:
        min_allowed = hi_f
    depth = np.where(np.isfinite(depth), depth, min_allowed)

    depth[depth <= 0.0] = min_allowed

    depth[depth < lo_f] = min_allowed

    depth_clipped = np.clip(depth, min_allowed, hi_f)

    if depth_bias != 0.0:
        depth_shifted = np.clip(depth_clipped + float(depth_bias), min_allowed, hi_f)
    else:
        depth_shifted = depth_clipped

    transformed = depth_curve_fn(depth_shifted)

    try:
        t_lo = float(depth_curve_fn(min_allowed + float(depth_bias)))
        t_hi = float(depth_curve_fn(hi_f))
    except Exception:
        t_lo = np.nanmin(transformed)
        t_hi = np.nanmax(transformed)

    t_min, t_max = min(t_lo, t_hi), max(t_lo, t_hi)

    if np.isclose(t_max, t_min):
        norm = np.zeros_like(transformed, dtype=np.float32)
    else:
        norm = (transformed - t_min) / (t_max - t_min)
        norm = np.clip(norm, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

    cmap = cm.get_cmap('turbo')
    colored = cmap(norm)[..., :3]

    if acc is not None:
        acc_arr = np.asarray(acc)
        if acc_arr.shape == norm.shape:
            colored = colored * acc_arr[..., None]

    return (colored * 255.0).astype(np.uint8)

def _compose_quad_canvas(
    gt_rgb,
    pred_rgb,
    dyn_col,
    depth_col,
    disp_h,
    disp_w,
    t,
    titles=("GT", "Prediction", "Dynamic Map", "Depth"),
    background_color=(255, 255, 255),
    outer_margin=20,
    gap_x=12,
    gap_y=12,
    time_area=28,
    title_area=22,
    space_title_to_video=6,
    space_time_to_title=6,
):

    def ensure_rgb(img):
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim == 2:
            return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        if a.ndim == 3 and a.shape[2] == 1:
            return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2RGB)
        if a.ndim == 3 and a.shape[2] == 4:
            return cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGBA2RGB)
        return a

    gt_rgb = ensure_rgb(gt_rgb)
    pred_rgb = ensure_rgb(pred_rgb)
    dyn_col = ensure_rgb(dyn_col)
    depth_col = ensure_rgb(depth_col)

    def shape_or_default(x, default_h, default_w):
        if x is None:
            return (default_h, default_w)
        return x.shape[:2]

    top0_h, top0_w = shape_or_default(gt_rgb, disp_h, disp_w)
    top1_h, top1_w = shape_or_default(pred_rgb, disp_h, disp_w)
    mid0_h, mid0_w = shape_or_default(dyn_col, disp_h, disp_w)
    mid1_h, mid1_w = shape_or_default(depth_col, disp_h, disp_w)  # careful: depth_col here is pred bottom_right when called; we treat symmetrically

    col0_w = max(top0_w, mid0_w)   # left column width
    col1_w = max(top1_w, mid1_w)   # right column width

    row0_h = max(top0_h, top1_h)   # top row height
    row1_h = max(mid0_h, mid1_h)   # mid row height
    row2_h = row1_h 
    content_w = col0_w + gap_x + col1_w
    content_h = (
        time_area
        + space_time_to_title
        + title_area
        + space_title_to_video
        + row0_h
        + gap_y
        + title_area
        + space_title_to_video
        + row1_h
        + gap_y
        + title_area
        + space_title_to_video
        + row2_h
    )

    canvas_w = int(max(1, content_w + outer_margin * 2))
    canvas_h = int(max(1, content_h + outer_margin * 2))
    canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)

    start_x = outer_margin + max(0, (canvas_w - outer_margin * 2 - content_w) // 2)
    start_y = outer_margin + max(0, (canvas_h - outer_margin * 2 - content_h) // 2)
    time_y_center = start_y + time_area // 2
    title0_center_y = start_y + time_area + space_time_to_title + title_area // 2
    row0_y = title0_center_y + title_area // 2 + space_title_to_video
    title1_center_y = row0_y + row0_h + gap_y + title_area // 2
    row1_y = title1_center_y + title_area // 2 + space_title_to_video
    title2_center_y = row1_y + row1_h + gap_y + title_area // 2
    row2_y = title2_center_y + title_area // 2 + space_title_to_video

    col0_x = start_x
    col1_x = start_x + col0_w + gap_x

    def paste_safe(src, canvas, y, x):
        if src is None:
            return
        src = np.asarray(src)
        if src.size == 0:
            return
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        elif src.ndim == 3 and src.shape[2] == 1:
            src = cv2.cvtColor(src[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif src.ndim == 3 and src.shape[2] == 4:
            src = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGBA2RGB)
        sh, sw = src.shape[:2]
        y0, x0 = int(y), int(x)
        y1 = y0 + sh
        x1 = x0 + sw
        # clip if out of canvas
        src_y0 = 0 if y0 >= 0 else -y0
        src_x0 = 0 if x0 >= 0 else -x0
        src_y1 = sh - max(0, y1 - canvas.shape[0])
        src_x1 = sw - max(0, x1 - canvas.shape[1])
        dst_y0 = max(y0, 0)
        dst_x0 = max(x0, 0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return
        if (src_y1 - src_y0) != sh or (src_x1 - src_x0) != sw:
            print(f"Warning: paste_safe clipped src ({sh}x{sw}) at ({y0},{x0}) to canvas ({canvas.shape[:2]}).")
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]
    paste_safe(gt_rgb, canvas, row0_y, col0_x)
    paste_safe(pred_rgb, canvas, row0_y, col1_x)
    # Row1 left/right  (dynamic)
    paste_safe(dyn_col if dyn_col is not None else None, canvas, row1_y, col0_x)
    paste_safe(None, canvas, row1_y, col1_x)  # placeholder - caller will pass pred_dyn separately when calling compose for tri-view case
    paste_safe(None, canvas, row2_y, col0_x)  # placeholder
    paste_safe(depth_col if depth_col is not None else None, canvas, row2_y, col1_x)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(min(col0_w, col1_w, row0_h, row1_h, row2_h) / 200))
    title_scale = max(0.5, min(col0_w, col1_w, row0_h, row1_h, row2_h) / 400.0)
    time_scale = title_scale * 1.1
    txt_color = (0, 0, 0)

    title_centers = [
        (col0_x + col0_w // 2, title0_center_y),
        (col1_x + col1_w // 2, title0_center_y),
        (col0_x + col0_w // 2, title1_center_y),
        (col1_x + col1_w // 2, title1_center_y),
    ]
    tnames = [titles[0], titles[1], "Dynamic", "Depth"]
    for (cx, ty), title_text in zip(title_centers, tnames):
        (tw, th), _ = cv2.getTextSize(title_text, font, title_scale, thickness)
        tx = int(cx - tw // 2)
        ty_text = int(ty + th // 2)
        cv2.rectangle(canvas, (tx - 6, ty_text - th - 4), (tx + tw + 6, ty_text + 4), background_color, -1)
        cv2.putText(canvas, title_text, (tx, ty_text), font, title_scale, txt_color, thickness, cv2.LINE_AA)

    counter_text = f"t={t}"
    (tw, th), _ = cv2.getTextSize(counter_text, font, time_scale, thickness + 1)
    cx_time = (canvas_w - tw) // 2
    cy_time = int(time_y_center + th // 2)
    cv2.rectangle(canvas, (cx_time - 6, cy_time - th - 6), (cx_time + tw + 6, cy_time + 6), background_color, -1)
    cv2.putText(canvas, counter_text, (cx_time, cy_time), font, time_scale, txt_color, thickness + 1, cv2.LINE_AA)

    return canvas

def _get_sky_mask_for_frame(sky_obj, idx, Ht, Wt):
    if sky_obj is None:
        return None
    try:
        if isinstance(sky_obj, torch.Tensor):
            arr = sky_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(sky_obj)
    except Exception:
        try:
            arr = np.asarray(sky_obj)
        except Exception:
            return None

    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr[idx]
        arr = arr[:, :, 0]
    elif arr.ndim == 3:
        arr = arr[idx]

    if arr.shape[:2] != (Ht, Wt):
        arr = cv2.resize(arr, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

    return arr

def _fit_to_display(img, Ht, Wt):
    if img is None:
        return np.zeros((Ht, Wt, 3), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.size == 0:
        return np.zeros((Ht, Wt, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    ih, iw = arr.shape[:2]
    Ht_safe = max(1, int(Ht))
    Wt_safe = max(1, int(Wt))
    if ih == Ht_safe and iw == Wt_safe:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    try:
        resized = cv2.resize(arr, (Wt_safe, Ht_safe), interpolation=cv2.INTER_LINEAR)
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)
        return resized
    except Exception:
        return np.zeros((Ht_safe, Wt_safe, 3), dtype=np.uint8)
import math
def make_comparison_video_quad(
    gt_frames,
    pred_frames,
    gt_dy_map,
    dyn_frames,
    gt_depth,
    depth_frames,
    sky_mask_frames,
    out_path,
    fps=8,
    views=3,
    titles=("GT", "Prediction", "Dynamic Map", "Depth"),
    cmap_name='plasma'
):
    import math, numpy as np, cv2, imageio, warnings, torch

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def fit_no_resize(img, H, W, bgcolor=(255,255,255)):
        if img is None:
            return np.ones((H, W, 3), dtype=np.uint8) * np.array(bgcolor, dtype=np.uint8)[None,None,:]
        a = np.asarray(img)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        elif a.ndim == 3 and a.shape[2] == 1:
            a = cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        sh, sw = a.shape[:2]
        if sh == H and sw == W:
            return a.astype(np.uint8)
        if sh >= H and sw >= W:
            y0 = (sh - H) // 2
            x0 = (sw - W) // 2
            return a[y0:y0+H, x0:x0+W].astype(np.uint8)
        if sh > H:
            y0 = (sh - H) // 2
            a = a[y0:y0+H, :, :]
            sh, sw = a.shape[:2]
        if sw > W:
            x0 = (sw - W) // 2
            a = a[:, x0:x0+W, :]
            sh, sw = a.shape[:2]
        canvas = np.ones((H, W, 3), dtype=np.uint8) * np.array(bgcolor, dtype=np.uint8)[None,None,:]
        y0 = (H - sh) // 2
        x0 = (W - sw) // 2
        canvas[y0:y0+sh, x0:x0+sw, :] = a.astype(np.uint8)
        return canvas

    def pad_to_multiple(img, multiple=16, bgcolor=(255,255,255)):
        img = np.asarray(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        new_h = int(math.ceil(h / multiple) * multiple)
        new_w = int(math.ceil(w / multiple) * multiple)
        if new_h == h and new_w == w:
            return img.astype(np.uint8)
        canvas = np.ones((new_h, new_w, 3), dtype=np.uint8) * np.array(bgcolor, dtype=np.uint8)[None,None,:]
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        canvas[y0:y0+h, x0:x0+w, :] = img.astype(np.uint8)
        return canvas

    def paste_safe(src, canvas, y, x):
        if src is None:
            return
        s = np.asarray(src)
        if s.ndim == 2:
            s = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
        if s.ndim == 3 and s.shape[2] == 1:
            s = cv2.cvtColor(s[:,:,0], cv2.COLOR_GRAY2RGB)
        if s.ndim == 3 and s.shape[2] == 4:
            s = s[:, :, :3]
        sh, sw = s.shape[:2]
        y0, x0 = int(y), int(x)
        y1 = min(canvas.shape[0], y0 + sh)
        x1 = min(canvas.shape[1], x0 + sw)
        src_y0 = 0 if y0 >= 0 else -y0
        src_x0 = 0 if x0 >= 0 else -x0
        src_y1 = src_y0 + (y1 - max(y0, 0))
        src_x1 = src_x0 + (x1 - max(x0, 0))
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return
        canvas[max(y0,0):y1, max(x0,0):x1] = s[src_y0:src_y1, src_x0:src_x1]

    # seq length and base sizes
    if isinstance(gt_frames, torch.Tensor):
        S = int(gt_frames.shape[0])
        _, _, H_init, W_init = gt_frames.shape
    else:
        arr = np.asarray(gt_frames)
        S = int(arr.shape[0])
        _, _, H_init, W_init = arr.shape

    disp_h = max(1, int(H_init))
    disp_w = max(1, int(W_init))
    spacer = 10
    pad_mult = 16

    cell_h = disp_h
    cell_w = (3 * disp_w + 2 * spacer) if views == 3 else disp_w

    # writers
    writer = None
    cv_writer = None
    try:
        writer = imageio.get_writer(out_path, fps=fps)
    except Exception as e:
        warnings.warn(f"imageio writer failed ({e}); will fallback to OpenCV later if needed", UserWarning)

    processed = []

    if views == 1:
        for t in range(S):
            # top RGB
            gt_rgb = _to_uint8_rgb(gt_frames[t]); pred_rgb = _to_uint8_rgb(pred_frames[t])
            gt_rgb = fit_no_resize(gt_rgb, cell_h, cell_w)
            pred_rgb = fit_no_resize(pred_rgb, cell_h, cell_w)

            # dyn
            gtd = to_np(gt_dy_map[t])
            if gtd.ndim == 3 and gtd.shape[2]==1: gtd = gtd[:,:,0]
            gt_dyn_col = _apply_colormap_gray(gtd, cmap_name=cmap_name)
            gt_dyn_col = fit_no_resize(gt_dyn_col, cell_h, cell_w)

            predd = to_np(dyn_frames[t])
            if predd.ndim == 3 and predd.shape[2]==1: predd = predd[:,:,0]
            pred_dyn_col = _apply_colormap_gray(predd, cmap_name=cmap_name)
            pred_dyn_col = fit_no_resize(pred_dyn_col, cell_h, cell_w)

            # depth
            gdep = to_np(gt_depth[t])
            if gdep.ndim == 3 and gdep.shape[2]==1: gdep = gdep[:,:,0]
            sky = _get_sky_mask_for_frame(sky_mask_frames, t, gdep.shape[0], gdep.shape[1])
            if sky is not None:
                m = np.asarray(sky).astype(bool)
                if m.shape != gdep.shape:
                    m = cv2.resize(m.astype(np.uint8), (gdep.shape[1], gdep.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                gdep[m] = 10.0
            gt_depth_col = visualize_depth(gdep, lo=0.10, hi=8.0, depth_curve_fn=lambda x: -np.log(x + 1e-6))
            gt_depth_col = fit_no_resize(gt_depth_col, cell_h, cell_w)

            pdep = to_np(depth_frames[t])
            if pdep.ndim == 3 and pdep.shape[2]==1: pdep = pdep[:,:,0]
            sky_p = _get_sky_mask_for_frame(sky_mask_frames, t, pdep.shape[0], pdep.shape[1])
            if sky_p is not None:
                m = np.asarray(sky_p).astype(bool)
                if m.shape != pdep.shape:
                    m = cv2.resize(m.astype(np.uint8), (pdep.shape[1], pdep.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                pdep[m] = 10.0
            pred_depth_col = visualize_depth(pdep, lo=0.10, hi=8.0, depth_curve_fn=lambda x: -np.log(x + 1e-6))
            pred_depth_col = fit_no_resize(pred_depth_col, cell_h, cell_w)

            processed.append({
                "top_left": gt_rgb, "top_right": pred_rgb,
                "mid_left": gt_dyn_col, "mid_right": pred_dyn_col,
                "bot_left": gt_depth_col, "bot_right": pred_depth_col,
                "t": t
            })

    else:  # views == 3
        groups = S // 3
        if S % 3 != 0:
            print(f"Warning: S={S} not divisible by 3, ignoring tail {S%3} frames.")
        white = np.ones((disp_h, spacer, 3), dtype=np.uint8) * 255
        for g in range(groups):
            i0 = 3*g; i1 = i0+1; i2 = i0+2
            # RGB tri-view
            gt0 = _to_uint8_rgb(gt_frames[i0]); gt1 = _to_uint8_rgb(gt_frames[i1]); gt2 = _to_uint8_rgb(gt_frames[i2])
            pr0 = _to_uint8_rgb(pred_frames[i0]); pr1 = _to_uint8_rgb(pred_frames[i1]); pr2 = _to_uint8_rgb(pred_frames[i2])
            gt0 = _fit_to_display(gt0, disp_h, disp_w); gt1 = _fit_to_display(gt1, disp_h, disp_w); gt2 = _fit_to_display(gt2, disp_h, disp_w)
            pr0 = _fit_to_display(pr0, disp_h, disp_w); pr1 = _fit_to_display(pr1, disp_h, disp_w); pr2 = _fit_to_display(pr2, disp_h, disp_w)
            gt_comp = np.concatenate([gt1, white, gt0, white, gt2], axis=1)  # disp_h x cell_w
            pr_comp = np.concatenate([pr1, white, pr0, white, pr2], axis=1)

            # dynamic tri-view (use full tri-view images as mid row; NO splitting)
            def get_dyn(idx):
                arr = to_np(gt_dy_map[idx])
                if arr.ndim==3 and arr.shape[2]==1: arr = arr[:,:,0]
                col = _apply_colormap_gray(arr, cmap_name=cmap_name)
                return _fit_to_display(col, disp_h, disp_w)
            def get_pred_dyn(idx):
                arr = to_np(dyn_frames[idx])
                if arr.ndim==3 and arr.shape[2]==1: arr = arr[:,:,0]
                col = _apply_colormap_gray(arr, cmap_name=cmap_name)
                return _fit_to_display(col, disp_h, disp_w)

            gt_dyn_comp = np.concatenate([get_dyn(i1), white, get_dyn(i0), white, get_dyn(i2)], axis=1)  # disp_h x cell_w
            pred_dyn_comp = np.concatenate([get_pred_dyn(i1), white, get_pred_dyn(i0), white, get_pred_dyn(i2)], axis=1)

            # depth tri-view (full tri-view as bottom row)
            def get_depth_comp(idx_source):
                c0 = to_np(idx_source[idx0 := 0]) if False else None  # dummy to satisfy linter
            # process depth for indices
            def process_depth_for(idx):
                dnp = to_np(gt_depth[idx])
                if dnp.ndim==3 and dnp.shape[2]==1: dnp = dnp[:,:,0]
                sky = _get_sky_mask_for_frame(sky_mask_frames, idx, dnp.shape[0], dnp.shape[1])
                if sky is not None:
                    m = np.asarray(sky).astype(bool)
                    if m.shape != dnp.shape:
                        m = cv2.resize(m.astype(np.uint8), (dnp.shape[1], dnp.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    dnp[m] = 10.0
                return visualize_depth(dnp, lo=0.10, hi=8.0, depth_curve_fn=lambda x: -np.log(x + 1e-6))
            def process_pred_depth_for(idx):
                dnp = to_np(depth_frames[idx])
                if dnp.ndim==3 and dnp.shape[2]==1: dnp = dnp[:,:,0]
                sky = _get_sky_mask_for_frame(sky_mask_frames, idx, dnp.shape[0], dnp.shape[1])
                if sky is not None:
                    m = np.asarray(sky).astype(bool)
                    if m.shape != dnp.shape:
                        m = cv2.resize(m.astype(np.uint8), (dnp.shape[1], dnp.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    dnp[m] = 10.0
                return visualize_depth(dnp, lo=0.10, hi=8.0, depth_curve_fn=lambda x: -np.log(x + 1e-6))

            # build depth tri-views (GT and Pred)
            gt_depth_l = process_depth_for(i1); gt_depth_c = process_depth_for(i0); gt_depth_r = process_depth_for(i2)
            gt_depth_l = _fit_to_display(gt_depth_l, disp_h, disp_w); gt_depth_c = _fit_to_display(gt_depth_c, disp_h, disp_w); gt_depth_r = _fit_to_display(gt_depth_r, disp_h, disp_w)
            gt_depth_comp = np.concatenate([gt_depth_l, white, gt_depth_c, white, gt_depth_r], axis=1)

            p_depth_l = process_pred_depth_for(i1); p_depth_c = process_pred_depth_for(i0); p_depth_r = process_pred_depth_for(i2)
            p_depth_l = _fit_to_display(p_depth_l, disp_h, disp_w); p_depth_c = _fit_to_display(p_depth_c, disp_h, disp_w); p_depth_r = _fit_to_display(p_depth_r, disp_h, disp_w)
            pred_depth_comp = np.concatenate([p_depth_l, white, p_depth_c, white, p_depth_r], axis=1)

            # Now fit_no_resize every composed tri-view to exactly (cell_h, cell_w)
            top_left = fit_no_resize(gt_comp, cell_h, cell_w)
            top_right = fit_no_resize(pr_comp, cell_h, cell_w)
            mid_left = fit_no_resize(gt_dyn_comp, cell_h, cell_w)
            mid_right = fit_no_resize(pred_dyn_comp, cell_h, cell_w)
            bot_left = fit_no_resize(gt_depth_comp, cell_h, cell_w)
            bot_right = fit_no_resize(pred_depth_comp, cell_h, cell_w)

            processed.append({
                "top_left": top_left, "top_right": top_right,
                "mid_left": mid_left, "mid_right": mid_right,
                "bot_left": bot_left, "bot_right": bot_right,
                "t": g
            })
    for item in processed:
        tl = item["top_left"]; tr = item["top_right"]
        ml = item["mid_left"]; mr = item["mid_right"]
        bl = item["bot_left"]; br = item["bot_right"]
        t = item["t"]

        col0_w = cell_w; col1_w = cell_w
        row0_h = cell_h; row1_h = cell_h; row2_h = cell_h

        gap_x = 12; gap_y = 12; outer_margin = 20
        time_area = 28; title_area = 22; space_time_to_title = 6; space_title_to_video = 6

        content_w = col0_w + gap_x + col1_w
        content_h = (
            time_area + space_time_to_title + title_area + space_title_to_video
            + row0_h + gap_y
            + title_area + space_title_to_video + row1_h + gap_y
            + title_area + space_title_to_video + row2_h
        )

        canvas_w = int(max(1, content_w + outer_margin*2))
        canvas_h = int(max(1, content_h + outer_margin*2))
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        start_x = outer_margin + max(0, (canvas_w - outer_margin*2 - content_w)//2)
        start_y = outer_margin + max(0, (canvas_h - outer_margin*2 - content_h)//2)

        time_y_center = start_y + time_area // 2
        title0_center_y = start_y + time_area + space_time_to_title + title_area // 2
        row0_y = title0_center_y + title_area // 2 + space_title_to_video
        title1_center_y = row0_y + row0_h + gap_y + title_area // 2
        row1_y = title1_center_y + title_area // 2 + space_title_to_video
        title2_center_y = row1_y + row1_h + gap_y + title_area // 2
        row2_y = title2_center_y + title_area // 2 + space_title_to_video

        col0_x = start_x
        col1_x = start_x + col0_w + gap_x

        # paste six cells
        paste_safe(tl, canvas, row0_y, col0_x)
        paste_safe(tr, canvas, row0_y, col1_x)
        paste_safe(ml, canvas, row1_y, col0_x)
        paste_safe(mr, canvas, row1_y, col1_x)
        paste_safe(bl, canvas, row2_y, col0_x)
        paste_safe(br, canvas, row2_y, col1_x)

        # draw titles: top row use titles[0]/titles[1]; mid/bot use GT/Pred prefix + titles[2]/titles[3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(min(col0_w, col1_w, row0_h) / 200))
        title_scale = max(0.45, min(col0_w, col1_w, row0_h) / 400.0)
        time_scale = title_scale * 1.0
        txt_color = (0,0,0)

        title_pairs = [
            (titles[0], titles[1]),
            (f"GT {titles[2]}", f"Pred {titles[2]}"),
            (f"GT {titles[3]}", f"Pred {titles[3]}"),
        ]
        centers = [
            (col0_x + col0_w // 2, title0_center_y),
            (col1_x + col1_w // 2, title0_center_y),
            (col0_x + col0_w // 2, title1_center_y),
            (col1_x + col1_w // 2, title1_center_y),
            (col0_x + col0_w // 2, title2_center_y),
            (col1_x + col1_w // 2, title2_center_y),
        ]
        texts = [title_pairs[0][0], title_pairs[0][1], title_pairs[1][0], title_pairs[1][1], title_pairs[2][0], title_pairs[2][1]]
        for (cx, cy), txt in zip(centers, texts):
            (tw, th), _ = cv2.getTextSize(txt, font, title_scale, thickness)
            tx = int(cx - tw//2)
            ty_text = int(cy + th//2)
            cv2.rectangle(canvas, (tx-6, ty_text-th-4), (tx+tw+6, ty_text+4), (255,255,255), -1)
            cv2.putText(canvas, txt, (tx, ty_text), font, title_scale, txt_color, thickness, cv2.LINE_AA)

        # time counter
        counter_text = f"t={t}"
        (tw, th), _ = cv2.getTextSize(counter_text, font, time_scale, thickness+1)
        cx_time = (canvas_w - tw)//2
        cy_time = int(time_y_center + th//2)
        cv2.rectangle(canvas, (cx_time-6, cy_time-th-6), (cx_time+tw+6, cy_time+6), (255,255,255), -1)
        cv2.putText(canvas, counter_text, (cx_time, cy_time), font, time_scale, txt_color, thickness+1, cv2.LINE_AA)

        # pad and write
        canvas_p = pad_to_multiple(canvas, pad_mult)

        # init cv_writer fallback if needed
        if writer is None and cv_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            cv_writer = cv2.VideoWriter(out_path, fourcc, float(fps), (canvas_p.shape[1], canvas_p.shape[0]))

        if writer is not None:
            writer.append_data(canvas_p)
        elif cv_writer is not None:
            try:
                bgr = cv2.cvtColor(canvas_p, cv2.COLOR_RGB2BGR)
                cv_writer.write(bgr)
            except Exception:
                pass
    if writer is not None:
        writer.close()
    if cv_writer is not None:
        cv_writer.release()
    return out_path
