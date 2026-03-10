FROM python:3.11.13-slim

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gdal-bin libgdal-dev gcc g++ procps python3-dev vim

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal 


ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install protobuf==3.20

# 5) App
EXPOSE 5050
CMD ["python","-u","app.py"]
