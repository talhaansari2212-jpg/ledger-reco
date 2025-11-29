import pandas as pd
import io

def export_report(matches, un_a, un_b, buffer):
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame({"Status": ["Done"], "Matches": [len(matches)]}).to_excel(writer, sheet_name="Summary", index=False)
        if not matches.empty:
            matches.to_excel(writer, sheet_name="Matched", index=False)
        if not un_a.empty:
            un_a.to_excel(writer, sheet_name="Unmatched_A", index=False)
        if not un_b.empty:
            un_b.to_excel(writer, sheet_name="Unmatched_B", index=False)
    buffer.seek(0)
