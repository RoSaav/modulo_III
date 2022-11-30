# server(backend)
FROM python:3.9

WORKDIR /server
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y vim

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]