from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from .utils import load_img, preprocess_img, postprocess_tens  # Ensure these are correctly implemented
from .eccv16 import eccv16  # Ensure the model class is correctly defined in eccv16.py
from .siggraph17 import SIGGRAPHGenerator  # Ensure SIGGRAPHGenerator is correctly implemented

app = FastAPI()

# Load the pretrained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eccv16_model = eccv16(pretrained=True).to(device).eval()
siggraph17_model = SIGGRAPHGenerator().to(device).eval()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Colorization API!"}

@app.get("/upload/", response_class=HTMLResponse)
async def upload_page():
    return """
    <html>
        <body>
            <h2>Upload an Image for Colorization</h2>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/upload/")
async def upload_and_colorize(file: UploadFile = File(...)):
    # Ensure the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read and process the image
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.asarray(img)
        tens_orig_l, tens_rs_l = preprocess_img(img_np)

        # Perform inference with ECCV16 model
        tens_rs_l = tens_rs_l.to(device)
        with torch.no_grad():
            out_ab_eccv16 = eccv16_model(tens_rs_l)

        # Perform inference with SIGGRAPH17 model
        with torch.no_grad():
            out_ab_siggraph17 = siggraph17_model(tens_rs_l)

        # Postprocess the outputs to generate the final colorized images
        colorized_img_eccv16 = postprocess_tens(tens_orig_l, out_ab_eccv16)
        colorized_img_siggraph17 = postprocess_tens(tens_orig_l, out_ab_siggraph17)

        # Convert images to PIL format for saving
        colorized_img_eccv16_pil = Image.fromarray((colorized_img_eccv16 * 255).astype(np.uint8))
        colorized_img_siggraph17_pil = Image.fromarray((colorized_img_siggraph17 * 255).astype(np.uint8))

        # Convert images to base64 for embedding in HTML
        original_img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        eccv16_buffer = BytesIO()
        colorized_img_eccv16_pil.save(eccv16_buffer, format="PNG")
        eccv16_buffer.seek(0)
        eccv16_base64 = base64.b64encode(eccv16_buffer.getvalue()).decode('utf-8')

        siggraph17_buffer = BytesIO()
        colorized_img_siggraph17_pil.save(siggraph17_buffer, format="PNG")
        siggraph17_buffer.seek(0)
        siggraph17_base64 = base64.b64encode(siggraph17_buffer.getvalue()).decode('utf-8')

        # Return a web page displaying the original and colorized images
        return HTMLResponse(f"""
        <html>
            <body>
                <h2>Original Image</h2>
                <img src="data:image/png;base64,{original_img_base64}" width="400"><br><br>
                <h2>ECCV16 Colorized Image</h2>
                <img src="data:image/png;base64,{eccv16_base64}" width="400"><br><br>
                <h2>SIGGRAPH17 Colorized Image</h2>
                <img src="data:image/png;base64,{siggraph17_base64}" width="400">
                <br><br>
                <a href="/upload/">Upload Another Image</a>
            </body>
        </html>
        """)

    except Exception as e:
        # Catch any exception and return an error message
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
