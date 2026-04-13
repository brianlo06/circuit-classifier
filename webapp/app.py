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
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from topology.analyze_symbol_with_proposals import analyze_with_proposals
from topology.pipeline import CircuitAnalysisPipeline
from topology.visualization import render_analysis, render_debug_analysis


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
TEMPLATES_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "static"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_DIR = PROJECT_ROOT / "data"
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
SAMPLE_SYMBOLS = [
    {
        "slug": "half_adder_vlabs",
        "label": "Half Adder VLAB",
        "description": "Supported symbol-style half adder with a clean compact layout.",
        "path": PROJECT_ROOT / "data" / "real_schematics" / "half_adder_vlabs.png",
        "image_url": "/data/real_schematics/half_adder_vlabs.png",
        "mode": "symbol_beta",
    },
    {
        "slug": "half_adder_gfg",
        "label": "Half Adder GFG",
        "description": "Supported colored symbol-style half adder from the current beta set.",
        "path": PROJECT_ROOT / "data" / "real_schematics" / "half_adder_gfg.jpg",
        "image_url": "/data/real_schematics/half_adder_gfg.jpg",
        "mode": "symbol_beta",
    },
    {
        "slug": "half_adder_tp",
        "label": "Half Adder TP",
        "description": "Supported symbol-style half adder with a tighter crop and small gate bodies.",
        "path": PROJECT_ROOT / "data" / "real_schematics" / "half_adder_tp.jpg",
        "image_url": "/data/real_schematics/half_adder_tp.jpg",
        "mode": "symbol_beta",
    },
    {
        "slug": "full_adder_uha",
        "label": "Full Adder UHA",
        "description": "Supported real full adder assembled from half adders.",
        "path": PROJECT_ROOT / "data" / "real_schematics" / "full_adder_using_half_adders.jpg",
        "image_url": "/data/real_schematics/full_adder_using_half_adders.jpg",
        "mode": "symbol_beta",
    },
    {
        "slug": "full_adder_tp",
        "label": "Full Adder TP",
        "description": "Supported promoted full-adder symbol case used as a regression guard.",
        "path": PROJECT_ROOT / "data" / "real_schematics" / "full_adder_tp.jpg",
        "image_url": "/data/real_schematics/full_adder_tp.jpg",
        "mode": "symbol_beta",
    },
]
MODE_COPY = {
    "fixture": {
        "status": "Stable",
        "summary": "Best for clean fixture-style schematics from the shipped demo path.",
        "tips": [
            "Use clean uploaded diagrams with boxed fixture-style gates.",
            "Stable scope is half_adder, half_subtractor, and full_adder.",
        ],
    },
    "symbol_beta": {
        "status": "Experimental",
        "summary": "Best for clean symbol-style gate drawings close to the current supported benchmark set.",
        "tips": [
            "Works best on colored or filled gate drawings with clear separation between gates and wires.",
            "Line-drawing textbook schematics are still a weak domain and may return unknown.",
            "Current promoted symbol benchmark: 6 supported real cases, including decoder_2to4.",
        ],
    },
}

app = FastAPI(title="Circuit Classifier Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
pipeline = CircuitAnalysisPipeline()


def _empty_context(request: Request) -> Dict[str, Any]:
    return {
        "request": request,
        "result": None,
        "error": None,
        "supported_circuits": ["half_adder", "half_subtractor", "full_adder"],
        "analysis_modes": [
            {
                "value": "fixture",
                "label": "Fixture Demo",
                "description": "Default stable path for clean fixture-style schematics.",
            },
            {
                "value": "symbol_beta",
                "label": "Symbol Beta",
                "description": "Experimental proposal-driven path for clean symbol-style gate drawings.",
            },
        ],
        "selected_mode": "fixture",
        "sample_fixtures": SAMPLE_FIXTURES,
        "sample_symbols": SAMPLE_SYMBOLS,
        "mode_copy": MODE_COPY,
    }


def _serialize_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = payload["result"]
    return {
        "mode": result["mode"],
        "classification": result["classification"],
        "gates": result["gates"],
        "selected_gate_ids": result.get("selected_gate_ids", []),
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


def _run_analysis(upload_path: Path, mode: str = "fixture") -> Dict[str, Any]:
    if mode == "symbol_beta":
        search = analyze_with_proposals(upload_path)
        result = search.result
        selected_gate_ids = list(search.selected_gate_ids)
        analysis_mode = "symbol_beta"
    else:
        result = pipeline.analyze(upload_path)
        selected_gate_ids = []
        analysis_mode = "fixture"

    warnings = list(result.warnings)
    if analysis_mode == "symbol_beta" and result.classification.label == "unknown":
        warnings.insert(
            0,
            "Symbol Beta could not match this upload to the current supported symbol-style set. "
            "Try one of the built-in symbol samples or use a clean colored gate diagram.",
        )

    analysis_image = render_analysis(result)
    debug_image = render_debug_analysis(result)

    return {
        "result": {
            "filename": upload_path.name,
            "mode": analysis_mode,
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
            "selected_gate_ids": selected_gate_ids,
            "warnings": warnings,
            "expressions": dict(result.classification.expressions),
            "truth_table": list(result.classification.truth_table),
            "analysis_image_url": _image_to_data_url(analysis_image),
            "debug_image_url": _image_to_data_url(debug_image),
        }
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _empty_context(request))


@app.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "model_path": str(pipeline.model_path)})


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.post("/analyze/sample", response_class=HTMLResponse)
async def analyze_sample(request: Request) -> HTMLResponse:
    context = _empty_context(request)
    form = await request.form()
    sample_slug = str(form.get("sample", "")).strip()
    requested_mode = str(form.get("mode", "fixture")).strip() or "fixture"
    template_version = str(form.get("template", "")).strip()
    template_name = "index_v2.html" if template_version == "v2" else "index.html"
    sample = next((item for item in SAMPLE_FIXTURES + SAMPLE_SYMBOLS if item["slug"] == sample_slug), None)
    if sample is None:
        context["error"] = f"Unknown sample: {sample_slug or 'missing'}"
        return templates.TemplateResponse(request, template_name, context, status_code=400)

    try:
        sample_mode = sample.get("mode", requested_mode if requested_mode in {"fixture", "symbol_beta"} else "fixture")
        context["selected_mode"] = sample_mode
        context.update(_run_analysis(sample["path"], mode=sample_mode))
        return templates.TemplateResponse(request, template_name, context)
    except Exception as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse(request, template_name, context, status_code=500)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    context = _empty_context(request)
    form = await request.form()
    requested_mode = str(form.get("mode", "fixture")).strip() or "fixture"
    template_version = str(form.get("template", "")).strip()
    template_name = "index_v2.html" if template_version == "v2" else "index.html"
    if requested_mode not in {"fixture", "symbol_beta"}:
        requested_mode = "fixture"
    context["selected_mode"] = requested_mode
    if not image.filename:
        context["error"] = "No file was provided."
        return templates.TemplateResponse(request, template_name, context, status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        context["error"] = f"Unsupported file type: {suffix or 'unknown'}"
        return templates.TemplateResponse(request, template_name, context, status_code=400)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = TMP_DIR / _normalize_filename(image.filename)

    try:
        with upload_path.open("wb") as output:
            shutil.copyfileobj(image.file, output)
        context.update(_run_analysis(upload_path, mode=requested_mode))
        return templates.TemplateResponse(request, template_name, context)
    except Exception as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse(request, template_name, context, status_code=500)
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
        payload = _run_analysis(upload_path, mode="fixture")
        return JSONResponse(_serialize_result(payload))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        image.file.close()
        upload_path.unlink(missing_ok=True)


@app.post("/api/analyze/symbol-beta", response_class=JSONResponse)
async def analyze_api_symbol_beta(image: UploadFile = File(...)) -> JSONResponse:
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
        payload = _run_analysis(upload_path, mode="symbol_beta")
        return JSONResponse(_serialize_result(payload))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        image.file.close()
        upload_path.unlink(missing_ok=True)
