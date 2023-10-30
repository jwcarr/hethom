Why do languages tolerate heterography? An experimental investigation into the emergence of informative orthography
===================================================================================================================

This repository contains data and code for our investigation of the emergence of heterographic homophones using iterated learning. The repo is organized into:

- `code/`: Python analysis code

- `data/`: JSON and CSV data files, and NetCDF model result archives

- `experiment/`: Node.js experimental code

- `manuscript/`: LaTeX manuscript and postscript figures


tl;dr
-----

- If you just want to get your hands on a CSV file, check `data/exp.csv`

- If you want to inspect the experiment code, check `experiment/server.js` and `experiment/client/client.js`

- If you want to look at the statistical models, check `code/model_exp1.py` and `code/model_exp2.py`


Replicating the analyses
------------------------

To dive into full replication, I would recommend that you first replicate my Python 3.11 environment. First, clone or download this repository and cd into the top-level directory:

```bash
$ cd path/to/hethom/
```

The exact version numbers of the Python packages I used are documented in `requirements.txt`. To replicate this environment and ensure that the required packages do not interfere with your own projects, create and activate a new Python virtual environment. Here's one way to do this:

```bash
$ python3 -m venv hethom_env
$ source hethom_env/bin/activate
```

With the new environment activated, install the required Python packages from `requirements.txt`:

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


Reproducable analysis pipeline
------------------------------

All intermediate and processed data files are included in this repo, so it it not necessary to reproduce all these steps unless you need to. The raw data pulled down from the database is located in `data/exp/`. This raw data went through the following pipeline:

RAW DATA FILES -> process_exp_data.py -> EXP.JSON -> build_csv.py -> EXP.CSV -> model_.py -> .NETCDF -> build_figures.py

- `process_exp_data.py` reduces the raw data into a single JSON file that contains only the most important information (essentially just the lexicon produced by each generation, arranged into conditions and chains). If you need to access other data, you may need to refer to the raw files themselves.

- `build_csv.py` takes the JSON output of the previous step and run various measures, most importantly communicative cost. These results are output to a CSV file for analysis.

- `model_*.py` scripts take in either a CSV or JSON file generated in the previous steps and fit the models to the data. The results are stored in NetCDF files.

- `build_figures.py` uses the data files generated in previous steps to create all the figures for the manuscript.


Experimental code
-----------------

The experiment itself is a Node.js web app, and is located under `experiment/`. If you just have some technical questions about the design, you may be able to find answers in `server.js` or `client.js`, which contain most of the experimental code. If you actually want to run the experiment, you will first need to install [Node.js](https://nodejs.org) and [MongoDB](https://www.mongodb.com) on your system/server. Once installed, `cd` into the experiment directory and install the required node modules:

```bash
$ cd experiment/
$ npm install
```

You will also need to make sure MongoDB is running, e.g.:

```bash
$ mkdir db
$ mongod -dbpath db
```

In `server.js`, set `PROTOCOL` to `http` (easier for testing) or `https` (secure), and set `PORT` to an open port number (you may need to open this port in a firewall). If you are using https, you will also need to provide the paths to the relevant encryption keys. If everything is set up correctly, you should be able to launch the server:

```bash
$ node server.js
```

In a browser, navigate to:

```
protocol://domain:port/?PROLIFIC_PID=000000000000000000000001
```

replacing protocol, domain, and port with the appropriate strings (e.g., `http://localhost:8080`). Initially, you will see "No task available", since no task has yet been added to the database. The tasks are defined in JSON files in `experiment/config/`. To launch one, run e.g.:

```bash
python mission_control.py exp launch
```

If the experiment has been launched successfully, it should be possible to get the status and open a chain:

```bash
python mission_control.py exp status
python mission_control.py exp open dif_lrn_0
```

Once a chain is open, you should be able to refresh the browser and run the experiment. Check `mission_control.py` for other things you can do.
