# Grayscale-Only Gate

Date: 2026-10-29

## Objective

Brain MRI scans are grayscale images. Color (RGB) uploads may be garbage
(non-MRI) input, so the API now rejects any image that carries real color.

## Design: content-based, not mode-based

A naive `mode != 'L'` check is wrong in practice: JPEG/PNG often store
genuine grayscale scans as three identical channels (`RGB` mode). The gate
below accepts all of those while rejecting anything with actual color:

- `api/inference.py::is_grayscale_image(image)`:
  - Modes `L/1/I/I;16/F/LA` → accepted directly.
  - Otherwise convert to RGB, downscale to ≤256px for speed, and compute the
    per-pixel channel spread `max(R,G,B) - min(R,G,B)`.
  - Accept if ≤1% of sampled pixels exceed a chroma spread of 15
    (tolerates JPEG/PNG compression noise); reject otherwise.
- `api/main.py::predict` raises `HTTPException(422)` with
  `COLOR_IMAGE_REJECTION_MESSAGE` before the engine ever sees the image
  (color uploads never reach the model).

Response example:

```json
{
  "error": "Rejected: this appears to be a color (RGB) image. The classifier only accepts grayscale brain MRI scans — color images are treated as potentially garbage input. Please upload a grayscale MRI scan (JPEG or PNG).",
  "detail": ""
}
```

## Frontend

- `frontend/src/lib/api.ts` now reads the `error` field (the backend's
  exception handler shape: `{"error": ..., "detail": ""}`) and throws a typed
  `ApiError` carrying the HTTP status.
- `frontend/src/routes/detector.tsx` no longer masks backend rejections with
  the demo simulation: HTTP rejections (e.g. color image, corrupt file) show a
  red banner with the backend message; only engine-unreachable network
  failures fall back to the simulated demo (now also flagged with an amber
  banner). The input checklist states "Grayscale scans only".

## Tests

`tests/test_api.py`:
- `test_predict_color_image_rejected` — colored PNG → 422, engine untouched.
- `test_predict_colorish_jpeg_rejected` — colored JPEG → 422, grayscale-content
  JPEG in an RGB container → 200.
- `test_is_grayscale_image_helper` — L/RGB-identical/colored/RGBA cases.
- `test_cors_health_endpoint` fixed to use a whitelisted origin
  (`http://localhost:5173`) — it previously asserted a header that the
  CORS middleware never emits for non-whitelisted origins; this failure
  predated the gate work.

Status: 25/25 `test_api.py` passed, 21/21 `test_preprocessing.py` passed.
