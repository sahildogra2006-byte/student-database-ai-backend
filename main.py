from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from database import create_db_and_tables
from routers.students import router as student_router
from routers.chatbot import router as chatbot_router


app = FastAPI(
    title="Student Database Application System",
    description="Modular FastAPI backend with CRUD APIs and an AI chatbot.",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    create_db_and_tables()


# ============================================================
# ROUTERS
# ============================================================

app.include_router(student_router)
app.include_router(chatbot_router)


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)


@app.get("/")
def root():
    return FileResponse(
        "frontend/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }