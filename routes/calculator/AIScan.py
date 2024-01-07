from flask import Blueprint

from calController.AIController import AIController

AIScan = Blueprint("AIScanner", __name__)

AIScan.route("/AIforApp", methods=["POST"])(AIController.AIforApp)