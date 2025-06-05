**How to run the model and api's for the deployment deadline:**
run:
docker build --no-cache -t movie-score-api .
docker run -d -p 8000:8000 movie-score-api  
go to:
<<<<<<< HEAD
http://localhost:8000/docs
click on try it out on the top right and then you can change the input json there. Finally click on execute to get the API's response.
=======

**How to run streamlit:**
run:
conda env create -f environment.yml
conda env create -f environment.yml
conda activate movieapi
streamlit run streamlit_app.py
>>>>>>> bfaae0f (finished streamlit implementation, still needs adjusting to the model inputs)
