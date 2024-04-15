FROM tensorflow/tensorflow:2.11.1

WORKDIR /usr/src/app
COPY requirements.txt ./

COPY farmingDataSetApi.py farmingDataSetApi.py
COPY potatoes.h5 potatoes.h5


EXPOSE 8000
CMD  pip install --no-cache-dir -r requirements.txt  && uvicorn  farmingDataSetApi:app --host 0.0.0.0 
# CMD [ "uvicorn", "farmingDataSetApi:app"," --host 0.0.0.0" ]
