FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# do stuff here
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY [ "Pipfile", "Pipfile.lock", "./"]
    

RUN pipenv install --system --deploy
COPY [ "starter.py",  "./"]

ENTRYPOINT [ "python", "starter.py", "yellow", "2022", "4"]