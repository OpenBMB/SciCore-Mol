import os
from html import escape
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse

UPSTREAM = os.environ.get("SCICORE_MOL_UPSTREAM", "").rstrip("/")
TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=30.0)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def status_page(title: str, detail: str) -> str:
    safe_title = escape(title)
    safe_detail = escape(detail)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SciCore-Mol Demo</title>
  <style>
    :root {{
      --ink: #10201d;
      --muted: #5f706b;
      --paper: #f4efe3;
      --edge: rgba(16, 32, 29, 0.16);
      --accent: #0f8b7d;
      --signal: #d66d2e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 12% 18%, rgba(15, 139, 125, 0.28), transparent 28rem),
        radial-gradient(circle at 86% 82%, rgba(214, 109, 46, 0.2), transparent 24rem),
        linear-gradient(135deg, #fbf6e9 0%, var(--paper) 48%, #e6efe9 100%);
      color: var(--ink);
      font-family: Charter, Georgia, serif;
    }}
    main {{
      width: min(880px, calc(100vw - 32px));
      border: 1px solid var(--edge);
      border-radius: 28px;
      padding: clamp(28px, 6vw, 64px);
      background: rgba(255, 252, 244, 0.74);
      box-shadow: 0 30px 90px rgba(16, 32, 29, 0.14);
      backdrop-filter: blur(18px);
    }}
    .eyebrow {{
      display: inline-flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 28px;
      color: var(--accent);
      font: 700 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }}
    .eyebrow::before {{
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--signal);
      box-shadow: 0 0 0 8px rgba(214, 109, 46, 0.13);
    }}
    h1 {{
      max-width: 720px;
      margin: 0;
      font-size: clamp(38px, 8vw, 82px);
      line-height: 0.9;
      letter-spacing: -0.055em;
    }}
    p {{
      max-width: 620px;
      margin: 26px 0 0;
      color: var(--muted);
      font-size: clamp(17px, 2vw, 21px);
      line-height: 1.65;
    }}
    code {{
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(15, 139, 125, 0.1);
      color: var(--accent);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.9em;
    }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">SciCore-Mol Demo</div>
    <h1>{safe_title}</h1>
    <p>{safe_detail}</p>
  </main>
</body>
</html>"""


def upstream_url(path: str, query: str) -> str:
    url = urljoin(f"{UPSTREAM}/", path)
    return f"{url}?{query}" if query else url


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request) -> Response:
    if not UPSTREAM:
        return HTMLResponse(
            status_page(
                "Demo endpoint not configured",
                "Set the Hugging Face Space secret SCICORE_MOL_UPSTREAM to the upstream demo origin.",
            ),
            status_code=503,
        )

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False) as client:
            upstream_response = await client.request(
                request.method,
                upstream_url(path, request.url.query),
                headers=headers,
                content=await request.body(),
            )
    except httpx.HTTPError as exc:
        return HTMLResponse(
            status_page("Demo is temporarily unreachable", str(exc)),
            status_code=502,
        )

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-type"
    }
    media_type = upstream_response.headers.get("content-type")

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=media_type,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
