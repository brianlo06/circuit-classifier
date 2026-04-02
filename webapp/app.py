"""
Minimal FastAPI web app for the circuit topology demo.
"""

import base64
import io
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from topology.pipeline import CircuitAnalysisPipeline
from topology.visualization import render_analysis, render_debug_analysis


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
TEMPLATES_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "static"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
TMP_DIR = PROJECT_ROOT / ".tmp" / "webapp_uploads"

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif"}
SAMPLE_FIXTURES = [
    {
        "slug": "half_adder",
        "label": "Half Adder",
        "description": "Two gates, clean baseline fixture.",
        "path": PROJECT_ROOT / "examples" / "schematics" / "half_adder.png",
        "image_url": "/examples/schematics/half_adder.png",
    },
    {
        "slug": "half_subtractor",
        "label": "Half Subtractor",
        "description": "Three gates with borrow logic and a NOT stage.",
        "path": PROJECT_ROOT / "examples" / "schematics" / "half_subtractor.png",
        "image_url": "/examples/schematics/half_subtractor.png",
    },
    {
        "slug": "full_adder_crossed",
        "label": "Full Adder",
        "description": "Five gates with deliberate orthogonal crossings.",
        "path": PROJECT_ROOT / "examples" / "schematics" / "full_adder_crossed.png",
        "image_url": "/examples/schematics/full_adder_crossed.png",
    },
]

app = FastAPI(title="Circuit Classifier Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
pipeline = CircuitAnalysisPipeline()


def _empty_context(request: Request) -> Dict[str, Any]:
    return {
        "request": request,
        "result": None,
        "error": None,
        "supported_circuits": ["half_adder", "half_subtractor", "full_adder"],
        "sample_fixtures": SAMPLE_FIXTURES,
    }


def _serialize_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = payload["result"]
    return {
        "classification": result["classification"],
        "gates": result["gates"],
        "warnings": result["warnings"],
        "expressions": result["expressions"],
        "truth_table": result["truth_table"],
    }


def _image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_filename(filename: str) -> str:
    safe_name = Path(filename or "upload.png").name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        suffix = ".png"
    return f"{uuid.uuid4().hex}{suffix}"


def _run_analysis(upload_path: Path) -> Dict[str, Any]:
    result = pipeline.analyze(upload_path)
    analysis_image = render_analysis(result)
    debug_image = render_debug_analysis(result)

    return {
        "result": {
            "filename": upload_path.name,
            "classification": {
                "label": result.classification.label,
                "confidence": result.classification.confidence,
                "reasoning": result.classification.reasoning,
            },
            "gates": [
                {
                    "gate_id": gate.gate_id,
                    "gate_type": gate.gate_type,
                    "confidence": gate.confidence,
                    "bbox": [gate.bbox.x1, gate.bbox.y1, gate.bbox.x2, gate.bbox.y2],
                }
                for gate in result.gates
            ],
            "warnings": list(result.warnings),
            "expressions": dict(result.classification.expressions),
            "truth_table": list(result.classification.truth_table),
            "analysis_image_url": _image_to_data_url(analysis_image),
            "debug_image_url": _image_to_data_url(debug_image),
        }
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", _empty_context(request))


@app.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "model_path": str(pipeline.model_path)})


@app.post("/analyze/sample", response_class=HTMLResponse)
async def analyze_sample(request: Request) -> HTMLResponse:
    context = _empty_context(request)
    form = await request.form()
    sample_slug = str(form.get("sample", "")).strip()
    sample = next((item for item in SAMPLE_FIXTURES if item["slug"] == sample_slug), None)
    if sample is None:
        context["error"] = f"Unknown sample fixture: {sample_slug or 'missing'}"
        return templates.TemplateResponse("index.html", context, status_code=400)

    try:
        context.update(_run_analysis(sample["path"]))
        return templates.TemplateResponse("index.html", context)
    except Exception as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse("index.html", context, status_code=500)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    context = _empty_context(request)
    if not image.filename:
        context["error"] = "No file was provided."
        return templates.TemplateResponse("index.html", context, status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        context["error"] = f"Unsupported file type: {suffix or 'unknown'}"
        return templates.TemplateResponse("index.html", context, status_code=400)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = TMP_DIR / _normalize_filename(image.filename)

    try:
        with upload_path.open("wb") as output:
            shutil.copyfileobj(image.file, output)
        context.update(_run_analysis(upload_path))
        return templates.TemplateResponse("index.html", context)
    except Exception as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse("index.html", context, status_code=500)
    finally:
        image.file.close()
        upload_path.unlink(missing_ok=True)


@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_api(image: UploadFile = File(...)) -> JSONResponse:
    if not image.filename:
        return JSONResponse({"error": "No file was provided."}, status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return JSONResponse({"error": f"Unsupported file type: {suffix or 'unknown'}"}, status_code=400)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = TMP_DIR / _normalize_filename(image.filename)

    try:
        with upload_path.open("wb") as output:
            shutil.copyfileobj(image.file, output)
        payload = _run_analysis(upload_path)
        return JSONResponse(_serialize_result(payload))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        image.file.close()
        upload_path.unlink(missing_ok=True)
