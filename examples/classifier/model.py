"""Stand-in for ANY ML model. In real use, load your artifact here (sklearn / xgboost / torch / onnx):

    import joblib; _m = joblib.load("model.pkl")
    def predict(x): return _m.predict([[x["income"], x["debt_ratio"], x["age"]]])[0]
"""
def predict(x):
    return "approve" if x["income"] > 40 and x["debt_ratio"] < 0.45 else "reject"
