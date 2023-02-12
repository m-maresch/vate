from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/detection")
async def detection(file: UploadFile):
    frame = await file.read()
    return {"file": f"{file.filename}"}
