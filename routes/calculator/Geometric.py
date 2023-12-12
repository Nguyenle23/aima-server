from flask import Blueprint

from calController.GeometricController import GeometricController

Geometric = Blueprint("Geometric", __name__)

Geometric.route("/square", methods=["POST"])(GeometricController.square)
Geometric.route("/rectangle", methods=["POST"])(GeometricController.rectangle)
Geometric.route("/triangle", methods=["POST"])(GeometricController.triangle)
# Optimize.route("/rectangle", methods=["POST"])(GeometricController)