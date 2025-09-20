"""Health check endpoint for Streamlit Cloud."""
from http import HTTPStatus
from flask import Flask, Response

app = Flask(__name__)

@app.route('/healthz')
def health_check():
    """Health check endpoint."""
    return Response("OK", status=HTTPStatus.OK, content_type='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
