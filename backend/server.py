import logging
import uvicorn

logging.basicConfig(level=logging.INFO)


def start_server():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    start_server()