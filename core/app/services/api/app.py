# app/services/api/app.py - Final update with all routers
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# Service instances (created here for lifespan management)
_ws_service = None
_monitoring_service = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - startup and shutdown events."""
    global _ws_service, _monitoring_service
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION} in {settings.ENVIRONMENT} mode")
    
    # Start WebSocket service
    from app.services.websocket.service import WebSocketService
    _ws_service = WebSocketService()
    await _ws_service.start()
    
    # Start monitoring service
    from app.services.monitoring.service import MonitoringService
    _monitoring_service = MonitoringService()
    await _monitoring_service.start()
    
    # Seed default admin for development
    if settings.ENVIRONMENT == "development":
        _seed_default_admin()
    
    yield
    
    # Shutdown
    if _ws_service:
        await _ws_service.stop()
    if _monitoring_service:
        await _monitoring_service.stop()
    
    logger.info(f"Shutting down {settings.APP_NAME}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="AgriMind AI Platform - Integrated agricultural intelligence",
        docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register all routers
    from app.services.health.router import router as health_router
    from app.services.auth.router import router as auth_router
    from app.services.datasets.router import router as datasets_router
    from app.services.pipeline.router import router as pipeline_router
    from app.services.ml.router import router as ml_router
    from app.services.feature_store.router import router as feature_store_router
    from app.services.inference.router import router as inference_router
    from app.services.reports.router import router as reports_router
    from app.services.monitoring.router import router as monitoring_router
    from app.services.websocket.router import router as websocket_router
    
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(datasets_router, prefix="/api")
    app.include_router(pipeline_router, prefix="/api")
    app.include_router(ml_router, prefix="/api")
    app.include_router(feature_store_router, prefix="/api")
    app.include_router(inference_router, prefix="/api")
    app.include_router(reports_router, prefix="/api")
    app.include_router(monitoring_router, prefix="/api")
    app.include_router(websocket_router, prefix="/api")
    
    return app

def _seed_default_admin():
    """Seed a default admin user for development."""
    import asyncio
    from app.services.auth.models import UserCreate, Role
    from app.services.auth.service import AuthService
    
    async def seed():
        try:
            await AuthService.create_user(UserCreate(
                username="admin",
                email="admin@agrimind.ai",
                password="admin123",  # Change in production!
                full_name="Admin User",
                role=Role.ADMIN,
            ))
            logger.info("Default admin user seeded")
        except ValueError:
            pass  # Already exists
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(seed())
        else:
            loop.run_until_complete(seed())
    except RuntimeError:
        asyncio.run(seed())
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="AgriMind AI Platform - Integrated agricultural intelligence",
        docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )
    
    # Register routers
    from app.services.health.router import router as health_router
    from app.services.auth.router import router as auth_router
    
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    
    # Will register more routers as we build them

    
    return app