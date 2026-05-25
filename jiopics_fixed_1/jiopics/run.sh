#!/bin/bash
# JIOPICS — Quick Start Script
echo "========================================="
echo "  JIOPICS — AI Collage Studio"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Install deps if needed
echo "Checking dependencies..."
pip install -r requirements.txt -q

echo ""
echo "Starting server..."
echo "Open: http://127.0.0.1:5000"
echo ""
cd backend && python app.py
