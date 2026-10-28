"""
ASDS — Automated Similarity Detection System
Development entry point.

Development:  python run.py
Production:   gunicorn run:app
"""

import os
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

from app import create_app

env = os.environ.get('FLASK_ENV', 'development')
config_name = 'development' if env == 'development' else 'production'
app = create_app(config_name)

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    print("Starting ASDS development server...")
    app.run(debug=(config_name == 'development'), host=host, port=port)
