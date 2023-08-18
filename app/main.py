from typing import Union
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
import logging
import time
from urllib.error import HTTPError, URLError
import polars as pl
import numpy as np 
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
import json
from datetime import datetime, timedelta
from pydantic import BaseModel
import io
import pyarrow.parquet as pa
import threading
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI(
    title='Actuarial Studio Pagination DataSet Service API',
    description="REST API for Actuarial Studio Pagination DataSet Service API",
    version="1.0.0")

class Connect_str(BaseModel):
    blob_connect_str: str = 'String connection'


def upload_to_blob(blob_service_client, container_name, filename_temp, df):
    logging.info("start to save the object")
    table = df.to_arrow()
    buffer = io.BytesIO()
    pa.write_table(table, buffer)
    parquet_bytes = buffer.getvalue()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename_temp)
    blob_client.upload_blob(io.BytesIO(parquet_bytes), overwrite=True)
    logging.info("finish to save the object")


def get_blob_url_with_token(blob_service_client, container_name, blob_name):
    permission = BlobSasPermissions(read=True, write=False, delete=False, create=False, add=False, update=False, process=False)
    expiry_time = datetime.utcnow() + timedelta(hours=1)
    token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=permission,
        expiry=expiry_time
    )
    url = f"{blob_service_client.primary_endpoint}{container_name}/{blob_name}?{token}"
    url = url.replace(" ", "%20")
    return url


def parse_from_df_to_json(df, offset, stop_after_rows,page_size,page_num):
    df = df[offset:stop_after_rows]
    json_data = df.write_json(row_oriented = True)

    logging.info(f"page_size {page_size}")
    logging.info(f"page_num {page_num}")

    parsed_json = {}
    parsed_json['columns'] = df.columns   
    parsed_json['rows'] = json.loads(json_data)
    parsed_json['currentpage'] = page_num
    return parsed_json


@app.post("/v1/paginate")
def get_data_paginated(
    blob_connect_obj: Connect_str,
    filename: str = Query('small.parquet', description="The name of the file to read"),
    page_num: int = Query(1, description="The page number"),
    page_size: int = Query(20, description="The number of rows per page"),
    sortColumn: str = Query(None, description="The column to sort by"),
    sortType: str = Query(None, description="The sort order"),
    search_column: str = Query(None, description="The column to search"),
    searchValue: str = Query(None, description="The value to search for")
):
    try:

        connect_str = blob_connect_obj.blob_connect_str
        logging.info(f"connect_str {connect_str}")

        offset = page_size * (page_num - 1)
        stop_after_rows = offset + page_size

        container_name = "deltatableparquet"
        container_name_temp = "deltatableparquettemp"


        if not filename:
            raise HTTPException(status_code=400, detail={"message":"File name is mandatory"})

        if not connect_str:
            raise HTTPException(status_code=400, detail={"message":"blob storage connection string is mandatory"})
        
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        if (searchValue and search_column) or sortColumn :
        
            filename_temp = filename.replace(".parquet", f"_{sortColumn}_{sortType}_{searchValue}_{search_column}.parquet")
            url = get_blob_url_with_token(blob_service_client, container_name_temp, filename_temp)
            logging.info(f"Full url with token for temporal file: {url}")    
            df = ''
            try:
                df = pl.read_parquet(url, n_rows=stop_after_rows, use_pyarrow=False,rechunk=False)
                parsed_json = parse_from_df_to_json(df, offset, stop_after_rows,page_size,page_num)
                logging.info('Python HTTP trigger function processed a request.')
                return JSONResponse(content=jsonable_encoder(parsed_json))
                
            except HTTPError as e:
                logging.info("the temporal file doesn't exist")
                logging.error(e, exc_info=True)
        else: 
            logging.info("enter into pagination without sort and search")
            url = get_blob_url_with_token(blob_service_client, container_name, filename)
            logging.info(f"Full url with token for Blob storage: {url}")
            df = ''
            try:
                df = pl.read_parquet(url, row_count_offset= offset,n_rows=stop_after_rows, use_pyarrow=False,rechunk=False)
                parsed_json = parse_from_df_to_json(df, offset, stop_after_rows,page_size,page_num)
                logging.info('Python HTTP trigger function processed a request.')
                return JSONResponse(content=jsonable_encoder(parsed_json))
            except HTTPError as e:
                logging.error(e, exc_info=True)
                raise HTTPException(status_code=400, detail={"message":e.reason})
            except URLError as e:
                logging.error(e, exc_info=True)
                raise HTTPException(status_code=400, detail={"message":"There was an error connecting to blob storage. Please check the connection string."})

        t1 = time.time()
        url = get_blob_url_with_token(blob_service_client, container_name, filename)
        logging.info(f"Full url with token for Blob storage: {url}")
        df = ''
        try:
            df = pl.read_parquet(url, use_pyarrow=False,rechunk=False)
        except HTTPError as e:
            logging.error(e, exc_info=True)
            raise HTTPException(status_code=400, detail={"message":e.reason})
        except URLError as e:
            logging.error(e, exc_info=True)
            raise HTTPException(status_code=400, detail={"message":"There was an error connecting to blob storage. Please check the connection string."})

        t2 = time.time()
        logging.info(("It takes %s seconds to download "+filename) % (t2 - t1))

        if searchValue and search_column:
            logging.info('Start to search value in the data file')
            if search_column not in df.columns:
                raise HTTPException(status_code=400, detail={"message":"search_column doesn't exist in the data file"})

            df = df.filter(df[search_column] == searchValue)
            logging.info('Finish searching value in the data file')

        if sortColumn: 
            logging.info('Start to sort the data file')
            if sortColumn not in df.columns:
                raise HTTPException(status_code=400, detail={"message":"SortColumn doesn't exist in the data file"})
            if sortType == "ascending":
                df = df.sort(by=[sortColumn], descending=False)
            else:
                df = df.sort(by=[sortColumn], descending=True)

        df_copy = df.clone()
        filename_temp = filename.replace(".parquet", f"_{sortColumn}_{sortType}_{searchValue}_{search_column}.parquet")
        t = threading.Thread(target=upload_to_blob, args=(blob_service_client, container_name_temp, filename_temp, df_copy))
        t.start()

        parsed_json = parse_from_df_to_json(df, offset, stop_after_rows,page_size,page_num)

        logging.info('Python HTTP trigger function processed a request.')

        return JSONResponse(content=jsonable_encoder(parsed_json))
    
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=400, detail={"message":"An error occurred while paginating the dataset"})

