**How to run the model and api's for the deployment deadline:**
run:
conda env create -f environment.yml
conda activate movieapi
uvicorn app:app --reload
go to:
http://127.0.0.1:8000/docs
click on try it out on the top right and then you can change the input json there. Finally click on execute to get the API's response.