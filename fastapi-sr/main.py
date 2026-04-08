import base64
import io
import os
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageSequence

from schemas import FileInfo, ProcessingOptions, TaskInfo, TaskStatus
import services.sr_model as sr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = Path("models")
MODEL_32_64 = MODELS_DIR / "model_32_64.pth"
MODEL_64_128 = MODELS_DIR / "model_64_128.pth"
MODEL_128_256 = MODELS_DIR / "model_128_256.pth"

for p in [MODEL_32_64, MODEL_64_128, MODEL_128_256]:
    if not p.exists():
        raise RuntimeError(f"Не найден файл модели: {p}")

g32_64 = sr.load_model(str(MODEL_32_64), DEVICE)
g64_128 = sr.load_model(str(MODEL_64_128), DEVICE)
g128_256 = sr.load_model(str(MODEL_128_256), DEVICE)

app = FastAPI(title="Arctic SAR SR")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

tasks: Dict[str, Dict] = {}


def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in [".png"]:
        return "png"
    if ext in [".jpg", ".jpeg"]:
        return "jpg"
    if ext in [".tif", ".tiff"]:
        return "tiff"
    return "png"


def extract_tiff_metadata(img: Image.Image):
    tag_v2 = getattr(img, "tag_v2", None)
    return tag_v2


def pil_to_preview(img: Image.Image, max_side: int = 800) -> Image.Image:
    preview = img.copy()
    preview.thumbnail((max_side, max_side))
    return preview


def pad_patch_reflect(arr: np.ndarray, top: int, left: int, patch_size: int) -> np.ndarray:
    h, w = arr.shape
    bottom = min(top + patch_size, h)
    right = min(left + patch_size, w)
    patch = arr[top:bottom, left:right]

    pad_h = patch_size - patch.shape[0]
    pad_w = patch_size - patch.shape[1]

    if pad_h > 0 or pad_w > 0:
        patch = np.pad(
            patch,
            ((0, pad_h), (0, pad_w)),
            mode="reflect"
        )
    return patch


def generate_positions(size: int, patch_size: int, overlap: int) -> List[int]:
    stride = patch_size - overlap
    positions = []
    pos = 0
    while pos < size:
        positions.append(pos)
        if pos + patch_size >= size:
            break
        pos += stride
        if pos + patch_size > size:
            pos = max(size - patch_size, 0)
    return sorted(set(positions))


def extract_patches(arr: np.ndarray, patch_size: int, overlap: int) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    h, w = arr.shape
    ys = generate_positions(h, patch_size, overlap)
    xs = generate_positions(w, patch_size, overlap)

    patches = []
    for y in ys:
        for x in xs:
            patch = pad_patch_reflect(arr, y, x, patch_size)
            patches.append(((y, x), patch))
    return patches


def make_weight_mask(size: int) -> np.ndarray:
    y = np.hanning(size) if size > 1 else np.array([1.0])
    x = np.hanning(size) if size > 1 else np.array([1.0])
    mask = np.outer(y, x).astype(np.float32)
    if mask.max() == 0:
        mask[:] = 1.0
    mask = np.clip(mask, 1e-3, None)
    return mask


def merge_patches(
    patches: List[Tuple[Tuple[int, int], np.ndarray]],
    out_h: int,
    out_w: int,
    patch_size: int
) -> np.ndarray:
    acc = np.zeros((out_h, out_w), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)
    mask = make_weight_mask(patch_size)

    for (y, x), patch in patches:
        y2 = min(y + patch_size, out_h)
        x2 = min(x + patch_size, out_w)
        ph = y2 - y
        pw = x2 - x

        acc[y:y2, x:x2] += patch[:ph, :pw] * mask[:ph, :pw]
        weights[y:y2, x:x2] += mask[:ph, :pw]

    weights = np.clip(weights, 1e-6, None)
    return acc / weights


def run_model_on_patch(patch01: np.ndarray, model) -> np.ndarray:
    t = sr.numpy01_to_tensor(patch01).to(DEVICE)
    with torch.no_grad():
        out = model(t)
    return sr.tensor_to_numpy01(out)


def process_stage_with_preview(
    arr01: np.ndarray,
    patch_size: int,
    overlap: int,
    model,
    task_id: str,
    file_idx: int,
    stage_name: str,
    global_progress_base: float,
    global_progress_span: float,
    fast_preview: bool
) -> np.ndarray:
    patches = extract_patches(arr01, patch_size, overlap)
    out_patch_size = patch_size * 2
    out_h = arr01.shape[0] * 2
    out_w = arr01.shape[1] * 2
    processed = []

    total = len(patches)
    for idx, ((y, x), patch) in enumerate(patches, start=1):
        out_patch = run_model_on_patch(patch, model)
        processed.append(((y * 2, x * 2), out_patch))

        if fast_preview:
            merged = merge_patches(processed, out_h, out_w, out_patch_size)
            preview_img = Image.fromarray((merged * 255).clip(0, 255).astype(np.uint8), mode="L")
            tasks[task_id]["results"][file_idx]["result_preview"] = image_to_data_url(pil_to_preview(preview_img), "PNG")
            tasks[task_id]["results"][file_idx]["current_stage"] = stage_name

        frac = idx / total
        tasks[task_id]["progress"] = min(
            99,
            int(global_progress_base + global_progress_span * frac)
        )

    merged = merge_patches(processed, out_h, out_w, out_patch_size)
    return merged


def save_image_bytes(img: Image.Image, fmt: str, original_tiff_tags=None, original_exif=None) -> bytes:
    buf = io.BytesIO()

    fmt_l = fmt.lower()
    if fmt_l == "png":
        img.save(buf, format="PNG")
    elif fmt_l in ["jpg", "jpeg"]:
        if img.mode != "L":
            img = img.convert("L")
        if original_exif:
            img.save(buf, format="JPEG", quality=95, exif=original_exif)
        else:
            img.save(buf, format="JPEG", quality=95)
    elif fmt_l in ["tif", "tiff"]:
        if original_tiff_tags is not None:
            img.save(buf, format="TIFF", tiffinfo=original_tiff_tags)
        else:
            img.save(buf, format="TIFF")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return buf.getvalue()


def process_image_pipeline(
    img: Image.Image,
    task_id: str,
    file_idx: int,
    fast_preview: bool,
    enable_retile: bool
) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")

    arr01 = np.array(img).astype(np.float32) / 255.0

    if enable_retile:
        stage1 = process_stage_with_preview(
            arr01, 32, 8, g32_64, task_id, file_idx, "32→64", 0, 30, fast_preview
        )
        stage2 = process_stage_with_preview(
            stage1, 64, 16, g64_128, task_id, file_idx, "64→128", 30, 30, fast_preview
        )
        stage3 = process_stage_with_preview(
            stage2, 128, 32, g128_256, task_id, file_idx, "128→256", 60, 35, fast_preview
        )
        final_arr = stage3
    else:
        patches = extract_patches(arr01, 32, 8)
        processed = []
        total = len(patches)
        final_patch_size = 256
        out_h = arr01.shape[0] * 8
        out_w = arr01.shape[1] * 8

        for idx, ((y, x), patch) in enumerate(patches, start=1):
            p1 = run_model_on_patch(patch, g32_64)
            p2 = run_model_on_patch(p1, g64_128)
            p3 = run_model_on_patch(p2, g128_256)
            processed.append(((y * 8, x * 8), p3))

            if fast_preview:
                merged = merge_patches(processed, out_h, out_w, final_patch_size)
                preview_img = Image.fromarray((merged * 255).clip(0, 255).astype(np.uint8), mode="L")
                tasks[task_id]["results"][file_idx]["result_preview"] = image_to_data_url(pil_to_preview(preview_img), "PNG")
                tasks[task_id]["results"][file_idx]["current_stage"] = "32→64→128→256 patch-wise"

            tasks[task_id]["progress"] = min(99, int(95 * idx / total))

        final_arr = merge_patches(processed, out_h, out_w, final_patch_size)

    final_img = Image.fromarray((final_arr * 255).clip(0, 255).astype(np.uint8), mode="L")
    return final_img


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/start", response_model=TaskInfo)
async def start_enhancement(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    fast_preview: bool = Form(False),
    enable_retile: bool = Form(True),
):
    if not files:
        raise HTTPException(status_code=400, detail="Не загружены файлы")

    task_id = str(uuid.uuid4())
    options = ProcessingOptions(
        fast_preview=fast_preview,
        enable_subsampling=enable_retile,
    )

    file_infos = []
    payload_files = []

    for f in files:
        raw = await f.read()
        if not raw:
            continue

        try:
            img = Image.open(io.BytesIO(raw))
            original_format = detect_format(f.filename)

            if img.mode != "L":
                img_l = img.convert("L")
            else:
                img_l = img.copy()

            w, h = img_l.size
            out_w, out_h = w * 8, h * 8

            file_infos.append(
                FileInfo(
                    filename=f.filename,
                    original_width=w,
                    original_height=h,
                    output_width=out_w,
                    output_height=out_h,
                )
            )

            payload_files.append({
                "filename": f.filename,
                "bytes": raw,
                "original_format": original_format,
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка чтения файла {f.filename}: {e}")

    tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "options": {
            "fast_preview": fast_preview,
            "enable_retile": enable_retile,
        },
        "results": [],
        "artifacts": {},
    }

    for info, pf in zip(file_infos, payload_files):
        img = Image.open(io.BytesIO(pf["bytes"]))
        preview = pil_to_preview(img.convert("L"))
        tasks[task_id]["results"].append({
            "filename": info.filename,
            "original_width": info.original_width,
            "original_height": info.original_height,
            "output_width": info.output_width,
            "output_height": info.output_height,
            "original_format": pf["original_format"],
            "allow_tiff": pf["original_format"] == "tiff",
            "original_preview": image_to_data_url(preview, "PNG"),
            "result_preview": None,
            "current_stage": "waiting",
        })

    background_tasks.add_task(
        run_enhancement_task,
        task_id,
        payload_files,
        fast_preview,
        enable_retile,
    )

    return TaskInfo(task_id=task_id, files=file_infos, options=options)


def run_enhancement_task(
    task_id: str,
    payload_files: List[Dict],
    fast_preview: bool,
    enable_retile: bool,
):
    try:
        total_files = len(payload_files)
        for idx, pf in enumerate(payload_files):
            img = Image.open(io.BytesIO(pf["bytes"]))
            original_tiff_tags = extract_tiff_metadata(img) if pf["original_format"] == "tiff" else None
            original_exif = img.info.get("exif") if hasattr(img, "info") else None

            final_img = process_image_pipeline(
                img=img,
                task_id=task_id,
                file_idx=idx,
                fast_preview=fast_preview,
                enable_retile=enable_retile,
            )

            tasks[task_id]["results"][idx]["result_preview"] = image_to_data_url(pil_to_preview(final_img), "PNG")
            tasks[task_id]["results"][idx]["current_stage"] = "completed"

            name = pf["filename"]
            tasks[task_id]["artifacts"][name] = {
                "png": save_image_bytes(final_img, "png", original_tiff_tags, original_exif),
                "jpg": save_image_bytes(final_img, "jpg", original_tiff_tags, original_exif),
            }

            if pf["original_format"] == "tiff":
                tasks[task_id]["artifacts"][name]["tiff"] = save_image_bytes(
                    final_img, "tiff", original_tiff_tags, original_exif
                )

            base_progress = int(((idx + 1) / total_files) * 100)
            tasks[task_id]["progress"] = min(100, base_progress)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    data = tasks[task_id]
    return {
        "task_id": task_id,
        "status": data["status"],
        "progress": data["progress"],
        "results": data["results"],
        "error": data.get("error"),
    }


@app.get("/download/{task_id}/file/{filename}/{fmt}")
async def download_single_file(task_id: str, filename: str, fmt: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    artifacts = tasks[task_id]["artifacts"]
    if filename not in artifacts:
        raise HTTPException(status_code=404, detail="File not found")
    if fmt not in artifacts[filename]:
        raise HTTPException(status_code=404, detail="Format not available")

    ext = "jpg" if fmt == "jpg" else fmt
    media = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "tiff": "image/tiff",
    }[fmt]

    stem = Path(filename).stem
    content = artifacts[filename][fmt]
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{stem}_sr.{ext}"'}
    )


def build_zip_for_task(task_id: str, mode: str) -> bytes:
    artifacts = tasks[task_id]["artifacts"]
    results = tasks[task_id]["results"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in results:
            filename = item["filename"]
            orig_format = item["original_format"]
            stem = Path(filename).stem

            if mode == "original":
                target_fmt = orig_format
                if target_fmt not in artifacts[filename]:
                    target_fmt = "png"
            else:
                target_fmt = mode
                if target_fmt not in artifacts[filename]:
                    continue

            ext = "jpg" if target_fmt == "jpg" else target_fmt
            zf.writestr(f"{stem}_sr.{ext}", artifacts[filename][target_fmt])

    buf.seek(0)
    return buf.getvalue()


@app.get("/download/{task_id}/all/{mode}")
async def download_all(task_id: str, mode: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    if mode not in ["original", "png", "jpg", "tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported mode")

    zip_bytes = build_zip_for_task(task_id, mode)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{task_id}_{mode}.zip"'}
    )
