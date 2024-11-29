import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI client
app = FastAPI()


# Create class with pydantic BaseModel
class InputRequest(BaseModel):
    input_str: str


def translate_text(input_str):
    completion = input_str + ": finished translation!"

    return completion


@app.post("/translate/")  # This line decorates 'translate' as a POST endpoint
async def translate(request: InputRequest):
    try:
        # Call your translation function
        translated_text = translate_text(request.input_str)
        return {"translated_text": translated_text}
    except Exception as e:
        # Handle exceptions or errors during translation
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8888, workers=1)
