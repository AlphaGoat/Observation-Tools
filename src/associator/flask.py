"""
Flask Server for Associatior service.

Author: Peter Thomas
Date: 21 October 2025
"""
from typing import List, Tuple
from flask import Flask, jsonify, request

from MHT import run_multiple_hypothesis_tracking

Obs = Tuple[float, float, float, float, int]

app = Flask("Associator")


class CustomException(Exception):
    pass


@app.errorhandler(CustomException)
def handle_exception(e):
    response = {
        'error': 'RuntimeException',
        'message': str(e)
    }
    return jsonify(response), 500


@app.route('/associateobs', methods=['GET'])
def associateobs(obs: List[List[Obs]], exposure_time: float, gap_time: float):
    """
    Given a list of observations over a set of frames,
    associate obs and form tracklets 
    """
    try:
        tracklets = run_multiple_hypothesis_tracking(obs, exposure_time, gap_time)
        if request.method == 'GET':
            return jsonify(tracklets)
    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Associatior Flask Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run application on.")
    parser.add_argument("--port", type=int, default=5050,
                        help="Port to run application on.")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)