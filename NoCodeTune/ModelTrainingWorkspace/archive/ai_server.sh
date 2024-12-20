#!/usr/bin/env bash

# download Python 3.10.6
VENVPATH="venv_textual_ai_model"
if [ ! -d "$VENVPATH" ]; then
    sudo apt-get update
    sudo apt-get install python3.10
    python3 -m venv  $VENVPATH
    source $VENVPATH/bin/activate
else
    # if folder already exists, just activate the virtual environment
    source $VENVPATH/bin/activate
fi


# Find the appropriate command to run Python. Check for 'python' and
# 'python3'. If neither are found, print an error message and exit.
function find_python_command() {
    if command -v python &> /dev/null
    then
        echo "python"
    elif command -v python3 &> /dev/null
    then
        echo "python3"
    else
        echo "Python not found. Please install Python."
        exit 1
    fi
}

PYTHON_CMD=$(find_python_command)

# Function to check if a port is available
    ## Finds the first available TCP port between 8090 and 10000.
    ## It does this by iterating through the range, checking if each port is
    ## currently in use by running "ss -tulwn | grep -q ":$PORT " and breaking
    ## out of the loop as soon as it finds an available port.
    ## The available port is then returned as output.
function find_free_port() {
    for PORT in $(seq 8090 10000); do
        ss -tulwn | grep -q ":$PORT " || break
    done
        echo $PORT
}

AVAILABLE_PORT=$(find_free_port)
echo $AVAILABLE_PORT
if $PYTHON_CMD -c "import sys; sys.exit(sys.version_info < (3, 10))"; then
    $PYTHON_CMD check_requirements.py requirements.txt
    if [ $? -eq 1 ]
    then 
        echo Installing missing packages...
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
    #$PYTHON_CMD -m server_file "$@"
    if [ "$SERVER_TYPE" = "data_load_server" ];then
        if [ "$NOHUP" = "True" ];then
            echo "nohub uvicorn activated"
            _DATE_="$(date '+%Y%m%d_%H%M%S')"
            nohup uvicorn DataLoad.data_load_server:app  --reload  --host 0.0.0.0 --port $AVAILABLE_PORT >> "ModelTrainingWorkspace/logs/data_load_server/${_DATE_}.out" 2>&1 &
        else
            echo "uvicorn activated"
        
            $PYTHON_CMD -m uvicorn DataLoad.data_load_server:app   --reload --host 0.0.0.0 --port $AVAILABLE_PORT 

        fi
        
    elif [ "$SERVER_TYPE" = "fine_tuning_server" ];then
        if [ "$NOHUP" = "True" ];then
            echo "nohub uvicorn activated"
            _DATE_="$(date '+%Y%m%d_%H%M%S')"
            nohup uvicorn fine_tuning_server:app  --reload  --host 0.0.0.0 --port $AVAILABLE_PORT >> "ModelTrainingWorkspace/logs/fine_tuning_server/${_DATE_}.out" 2>&1 &
        else
            echo "uvicorn activated"
        
            $PYTHON_CMD -m uvicorn fine_tuning_server:app   --reload --host 0.0.0.0 --port $AVAILABLE_PORT
        
        fi

    else
        echo "TASK_TYPE must be one of the data_load_server, fine_tuning_server, model_release_server "
    
    fi
        read -p "Press any key to continue..."
else
    echo "Python 3.10 or higher is required to release server."
fi
