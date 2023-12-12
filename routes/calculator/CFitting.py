from flask import Blueprint

from calController.CFittingController import CFittingController

CFitting = Blueprint("CFitting", __name__)

CFitting.route("/lRegression", methods=["POST"])(CFittingController.linearRs)
CFitting.route("/secOrderLR", methods=["POST"])(CFittingController.secOrderLR)
CFitting.route("/mulOrderLR", methods=["POST"])(CFittingController.multiOrderLR)