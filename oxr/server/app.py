import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from oxr.config.settings import settings
from oxr.model.client import model_client
from oxr.model.layout_client import layout_client
from oxr.server.queue.memory_store import MemoryTaskStore
from oxr.server.demo_api import router as demo_router, start_demo, stop_demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_for_server()
    app.state.task_store = MemoryTaskStore()
    await start_demo(app)
    try:
        yield
    finally:
        demo_close_results = await asyncio.gather(stop_demo(app), return_exceptions=True)
        close_results = await asyncio.gather(
            layout_client.aclose(),
            model_client.aclose(),
            return_exceptions=True,
        )
        for result in demo_close_results + close_results:
            if isinstance(result, BaseException):
                raise result


app = FastAPI(title="OXR Server", version="0.1.0", lifespan=lifespan)

from oxr.server.routes import router

app.include_router(router, prefix="/v1")

# OXR Online Demo — bundled UI and its dedicated API endpoints.
app.include_router(demo_router, prefix="/v1")
DEMO_DIR = Path(__file__).parent / "demo"


@app.get("/demo", include_in_schema=False)
@app.get("/demo/", include_in_schema=False)
async def demo_page():
    return FileResponse(DEMO_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/demo/assets", StaticFiles(directory=DEMO_DIR, check_dir=False), name="demo-assets")
