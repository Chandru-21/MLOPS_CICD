FROM python:3.10-slim-buster

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

WORKDIR /code

#copy to code directory
COPY . /code 

# #set permissions

RUN chmod +x /code/prediction_model

RUN chmod +w /code/prediction_model/trained_models


ENV PYTHONPATH "${PYTHONPATH}:/code/prediction_model"

# RUN pip install --no-cache-dir --upgrade -r code/src/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN python /code/prediction_model/training_pipeline.py

EXPOSE 8005

# WORKDIR /code/src

# ENV PYTHONPATH "${PYTHONPATH}:/code/src"

ENTRYPOINT ["python"]

CMD ["main.py"]


# CMD pip install -e .