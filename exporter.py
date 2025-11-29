import pandas as pd
import io
import joblib
import os

def export_report(matches, un_a, un_b, buffer, include_model_info=True, model_path="match_model.joblib"):
    """
    Writes an Excel report to the given buffer (BytesIO).
    Adds a Summary sheet, Matched, Unmatched_A, Unmatched_B and optional Model Info sheet.
    """
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary
        summary = {
            "Status": ["Done"],
            "Matches": [len(matches) if matches is not None else 0],
            "Unmatched_A": [len(un_a) if un_a is not None else 0],
            "Unmatched_B": [len(un_b) if un_b is not None else 0],
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        # Matched
        if matches is not None and not matches.empty:
            matches.to_excel(writer, sheet_name="Matched", index=False)

        # Unmatched
        if un_a is not None and not un_a.empty:
            un_a.to_excel(writer, sheet_name="Unmatched_A", index=False)
        if un_b is not None and not un_b.empty:
            un_b.to_excel(writer, sheet_name="Unmatched_B", index=False)

        # Model info / audit
        if include_model_info and os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                info = {
                    "Model": ["RandomForestClassifier"],
                    "Path": [model_path],
                    "N_Estimators": [getattr(model, "n_estimators", "")],
                    "Trained": [True]
                }
                pd.DataFrame(info).to_excel(writer, sheet_name="Model_Info", index=False)
            except Exception:
                # fallback sheet
                pd.DataFrame({"Model": ["Not available"]}).to_excel(writer, sheet_name="Model_Info", index=False)

    buffer.seek(0)
