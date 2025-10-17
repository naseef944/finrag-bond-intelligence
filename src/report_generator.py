import json
from datetime import datetime

def build_report(forecast, rag_summary, top_drivers=None):
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "forecast": forecast,
        "rag_summary": rag_summary,
        "top_drivers": top_drivers or []
    }
    return report

if __name__ == "__main__":
    print(json.dumps(build_report({"next_q":200}, "Stable market", [{"var":"inflation","effect":"up"}]), indent=2))

