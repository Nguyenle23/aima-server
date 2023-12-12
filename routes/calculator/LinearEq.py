from flask import Blueprint

from calController.LinearEqController import LinearEqController

LinearEq = Blueprint("Linear", __name__)

LinearEq.route("/naivege", methods=["POST"])(LinearEqController.NaiveGE)