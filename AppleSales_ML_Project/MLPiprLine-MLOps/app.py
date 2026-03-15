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
from src.pipline.prediction_pipeline import AppleRetailSalesData, AppleRetailSalesDataRegressor
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
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Product_Name: Optional[int] = None
        self.quantity: Optional[int] = None
        self.Store_Name: Optional[int] = None
        self.City: Optional[float] = None
        self.category_name: Optional[int] = None
        self.sale_year: Optional[float] = None
        self.sale_month: Optional[float] = None
        self.Launch_Year: Optional[float] = None
        self.Laucnh_Month: Optional[float] = None

    async def get_Apple_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Product_Name = form.get("Product_Name")
        self.quantity = form.get("quantity")
        self.Store_Name = form.get("Store_Name")
        self.City = form.get("City")
        self.category_name = form.get("category_name")
        self.sale_year = form.get("sale_year")
        self.sale_month = form.get("sale_month")
        self.Launch_Year = form.get("Launch_Year")
        self.Laucnh_Month = form.get("Launch_Month")
        

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "appledata.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_Apple_data()
        
        apple_data = AppleRetailSalesData(
                                Product_Name = form.Product_Name,
                                quantity = form.quantity,
                                Store_Name = form.Store_Name,
                                City = form.City,
                                category_name = form.category_name,
                                sale_year = form.sale_name,
                                sale_month = form.sale_month,
                                Launch_Year = form.Launch_Year,
                                Launch_Month = form.Laucnh_Month
                                )

        # Convert form data into a DataFrame for the model
        apple_df = apple_data.get_vehicle_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = AppleRetailSalesDataRegressor()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=apple_df)[0]

        output = round(value[0],2)
        return {'Total_Sale is {}'.format(output)}

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "appledata.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"value": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)