# Physical Protection System (PPS) Facial Recognition System (FRS)

## Description and Purpose

This is a Flask server designed to match faces to IDs from a database.

The server takes an image and compares its encodeing to the stored encodings in the database and returns the ID of the best matched face, along with a (scaled) confidence score.

## Installation Instructions 

**Python Version:** `3.8.10`

**APT Extras:** None

**PIP Packages:**
```bash
pip install requirements.txt
```

## Directory Structure

`conf` - folder containing a data file for predicting facial landmarks for encoding faces

`db` - folder containing sqlite3 databases

`faces` - folder of images for each face, which are inserted into the database

`log` - folder containing log files

`tmp` - test folder, can be ignored

## Starting the Server

```bash
python3 frs.py
```
