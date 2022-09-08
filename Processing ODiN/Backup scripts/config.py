# %% Imports
import logging
from sqlalchemy import create_engine, engine
from pathlib import Path
import os
import sys
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.insert(1, parent_directory)
import credentials as cred

debug = False
export_to_csv = True

# Set the project root folder
root = Path.cwd()#.parent
sourcedata = Path(root, "raw")
data_out = Path(root, "output")

# %% Logger
# Create logger
logger = logging.getLogger("obs_sched")
consoleHandler = logging.StreamHandler()
if debug is True:
    loglevel = logging.DEBUG
else:
    loglevel = logging.INFO
# Set loglevel
consoleHandler.setLevel(loglevel)
logger.setLevel(loglevel)

# Create and add formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
consoleHandler.setFormatter(formatter)

# Add the consolehandler, but check if there's a handler already.
if len(logger.handlers) == 0:
    logger.debug("Adding new log handler, none exists yet.")
    logger.addHandler(consoleHandler)
# %% Files
wogem_pickle = Path(sourcedata, "wogem.pkl")
pc4_mrdh_pickle = Path(sourcedata, "pc4_mrdh.pkl")
odin_debug_file = Path(sourcedata, "odin_debug.pkl")
odin_file = Path(sourcedata, "odin.pkl")

# %% SQL connection
# Create an ODiN and GTFS db engine
odin_url = engine.URL.create(
    "postgresql+psycopg2",
    username=cred.db1_user,
    password=cred.db1_pass,
    host=cred.db1_serv,
    port=cred.db1_port,
    database=cred.db1_name,
)

odin_engine = create_engine(odin_url)
