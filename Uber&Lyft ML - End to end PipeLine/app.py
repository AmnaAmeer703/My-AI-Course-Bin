from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import UberLyftData, UberLyftDataRegressor
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.distance: Optional[int] = None
        self.cab_type: Optional[int] = None
        self.destination: Optional[int] = None
        self.source: Optional[float] = None
        self.serge_multiplier: Optional[int] = None
        self.name: Optional[float] = None
        self.Period_Of_Time: Optional[float] = None
                

    async def get_uberlyft_data(self):
        
        form = await self.request.form()
        self.distance = form.get("distance")
        self.cab_type = form.get("cab_type")
        self.destination = form.get("destination")
        self.source = form.get("source")
        self.serge_multiplier = form.get("serge_multiplier")
        self.name = form.get("name")
        self.Period_Of_Time = form.get("Period_Of_Time")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    
    return templates.TemplateResponse(
            "uberlyft.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    
    try:
        form = DataForm(request)
        await form.get_uberlyft_data()
        
        uberlyft_data = UberLyftData(
                                distance= form.distance,
                                cab_type = form.cab_type,
                                destination = form.destination,
                                source = form.source,
                                serge_multiplier = form.serge_multiplier,
                                name = form.name,
                                Period_Of_Time = form.Period_Of_Time
                                )

        # Convert form data into a DataFrame for the model
        uberlyft_df = uberlyft_data.get_uberlyft_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = UberLyftDataRegressor()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=uberlyft_df)[0]

        output = round(value[0],2)
        return {'Charges are {}'.format(output)}


        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "uberlyft.html",
            {"request": request, "context": output},
        )
    except Exception as e:
        return {"error": f"{e}"}


# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)