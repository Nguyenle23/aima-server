from flask import Blueprint

from calController.CalculusController import CalculusController

Calculus = Blueprint("Calculus", __name__)

Calculus.route("/fundamental", methods=["POST"])(CalculusController.fundamental)
Calculus.route("/linearalgebra", methods=["POST"])(CalculusController.LinearAlgebra)