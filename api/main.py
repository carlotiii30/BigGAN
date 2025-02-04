from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.endpoints import image_endpoints, model_endpoints

app = FastAPI()


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


app.include_router(model_endpoints.router, prefix="/models", tags=["models"])
app.include_router(image_endpoints.router, prefix="/images", tags=["images"])
