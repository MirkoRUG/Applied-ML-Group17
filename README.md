**How to run the model with docker**
run:
docker compose up --develop
http://localhost:8501/
**If you wannt to run this with conda instead of docker for any reason**
run:
conda env create -f environment.yml
conda activate movieapi
streamlit run main.py
