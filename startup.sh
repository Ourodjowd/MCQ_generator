#!/bin/bash
# This script is used to start the Streamlit application
# It sets the environment variable and runs the Streamlit command 0.0.0.0
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
chmod +x startup.sh 