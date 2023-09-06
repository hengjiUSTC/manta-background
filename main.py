from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from background import inference  # Make sure to import your inference function here
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    docs_url="/jhdc"
)
origins = [
    "http://localhost:5173",
    "https://mantaverse.xyz"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process/")
async def process_image(file: UploadFile):
    # Read image directly into memory
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Save image to temporary BytesIO object for inference
    img_io = BytesIO()
    image.save(img_io, format='JPEG')
    img_io.seek(0)
    
    # Use your inference function
    im_rgba, pil_mask = inference(img_io)  # Update your inference function to accept BytesIO or PIL Image
    
    # Convert output image to Bytes
    output_img_io = BytesIO()
    im_rgba.save(output_img_io, format='PNG')
    output_img_io.seek(0)
    
    return StreamingResponse(output_img_io, media_type="image/png", headers={"Content-Disposition": "attachment; filename=processed_image.png"})
