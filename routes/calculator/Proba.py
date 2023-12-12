from flask import Blueprint

from calController.ProbaController import ProbaController

Proba = Blueprint("Probability", __name__)

Proba.route("/probaCal", methods=["POST"])(ProbaController.Probability)
Proba.route("/probaIndepend", methods=["POST"])(ProbaController.ProbaIndepend)
Proba.route("/normalDis", methods=["POST"])(ProbaController.NormalDistribution)
Proba.route("/statistic", methods=["POST"])(ProbaController.Statistic)
Proba.route("/conInterval", methods=["POST"])(ProbaController.ConInterval)
Proba.route("/sampleSize", methods=["POST"])(ProbaController.SampleSize)
Proba.route("/zScore", methods=["POST"])(ProbaController.ZScore)