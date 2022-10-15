from fastapi import FastAPI
import names
from enum import Enum
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, confloat


app = FastAPI()

# Path operation decorator
@app.get("/")
# Path operation function
async def root():
    return {"message": "Hello World"}

@app.get("/random_name")
async def root():
    return {"message": f"Hello {names.get_first_name()}"}

song_db = ["evermore", "willow", "champagne problems", "gold rush",
           "'tis the damn season'", "tolerate it", "no body no crime",
           "happiness", "dorothea", "coney island",
           "ivy", "cowboy like me", "long story short"]


@app.get("/name/andre")
async def get_name():
    return {"message": f"Hello Hans"}

@app.get("/name/{name}")
async def get_name(name):
    return {"message": f"Hello {name}"}


@app.get("/songs/{song_id}")
async def get_song(song_id: int):
    return {"song_id": song_id, "song_name": song_db[song_id]}


class AlbumName(str, Enum):
    folklore = "folklore"
    evermore = "evermore"
    red = "red"

@app.get("/album/{album_name}")
async def get_album(album_name: AlbumName):
    if album_name == AlbumName.folklore:
        return {"album_name": album_name, "year": 2020}
    if album_name.value == "evermore":
        return {"album_name": album_name, "year": 2020}
    return {"album_name": album_name, "year": 2021}

@app.get("/song_list")
async def get_song_list(skip: int = 0, limit: int = 10):
    return song_db[skip: skip + limit]


class Album(BaseModel):
    name: str
    description: str = None   
    price: confloat(gt=0) = 'No description given'
    discount:float = None


@app.post("/albums/")
async def create_album(album: Album):
    album_dict = album.dict()
    ## save album
    return album_dict