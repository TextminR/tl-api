FROM python:3.10 AS build

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake
RUN pip install --user --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --user --no-cache-dir --upgrade -r /requirements.txt

FROM python:3.10
WORKDIR /app

COPY --from=build /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY ./app .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
EXPOSE 80