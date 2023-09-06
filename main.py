from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from background import inference
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(docs_url="/jhdc")
origins = ["http://localhost:5173", "https://mantaverse.xyz"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    # Perform inference
    im_rgba, _ = inference(image)

    # Convert to byte stream
    img_io = BytesIO()
    im_rgba.save(img_io, "PNG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")
