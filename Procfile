web: cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
worker: cd backend && python3 worker.py
