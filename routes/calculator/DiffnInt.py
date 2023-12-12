from flask import Blueprint

from calController.DiffnIntController import DiffnIntController

DiffnInt = Blueprint("DiffnInt", __name__)

DiffnInt.route("/trapezoidal", methods=["POST"])(DiffnIntController.trapezoidal)
DiffnInt.route("/trapezoidalMA", methods=["POST"])(DiffnIntController.trapezoidalMA)