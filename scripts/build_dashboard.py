"""Build reports/caresignal_dashboard.xlsx: an Excel monitoring dashboard for the readmission model.

Sheets: Summary (KPIs), Outreach (capacity what-if), Calibration, Fairness, Drift, Segments.
Raw counts come from the registered champion model scored on the gold test split; every rate,
lift, gap and status in the workbook is an Excel formula on those counts, with a native chart.

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
HEAD = PatternFill("solid", start_color="1F4E79")
INPUT = Font(name=FONT, color="0000FF", bold=True)
THIN = Side(style="thin", color="BFBFBF")


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def body_font(ws):
    for row in ws.iter_rows():
        for cell in row:
            if cell.font.name != FONT:
                f = copy(cell.font)
                f.name = FONT
                cell.font = f
            if cell.value is not None:
                cell.border = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)


def widths(ws, w):
    for i, x in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(i)].width = x


def tidy(chart, title, xt, yt):
    chart.title, chart.x_axis.title, chart.y_axis.title = title, xt, yt
    chart.roundedCorners = False
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    if chart.legend is not None:
        chart.legend.overlay = False
    chart.height, chart.width = 8, 16


def main() -> None:
    gold = pd.read_parquet("lakehouse/gold/readmission_features")
    te = gold[gold.split.astype(str) == "test"].reset_index(drop=True)
    model = mlflow.pyfunc.load_model("models:/caresignal-readmission@champion")
    p = np.asarray(model.predict(te))
    y = te.label.to_numpy()
    res = json.loads(Path("reports/readmission_results.json").read_text())
    wb = Workbook()

    # ---- Summary ----
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "CareSignal: 30-day readmission model, held-out test patients"
    ws["A1"].font = Font(name=FONT, bold=True, size=14, color="1F4E79")
    ws["A2"] = "Source: reports/readmission_results.json and the registered champion model scored on the gold test split."
    rows = [("Metric", "Value", "Note"),
            ("Test encounters", len(te), "patients never seen in training"),
            ("30-day readmissions", int(y.sum()), ""),
            ("Readmission rate", "=B5/B4", "formula"),
            ("LightGBM + isotonic AUROC", res["test"]["auroc"], "champion"),
            ("AUROC 95% CI low", res["test"]["auroc_95ci"][0], "bootstrap, 1,000 resamples"),
            ("AUROC 95% CI high", res["test"]["auroc_95ci"][1], ""),
            ("Spark MLlib baseline AUROC", res["baseline_spark_lr"]["test_auroc"], "logistic regression"),
            ("AUROC gain over baseline", "=B7-B10", "formula"),
            ("AUPRC", res["test"]["auprc"], ""),
            ("Brier score", res["test"]["brier"], "lower is better"),
            ("Calibration error (ECE)", res["test"]["ece"], "after isotonic calibration"),
            ("Score drift PSI (train → test)", res["drift"]["score_psi"], "<0.10 stable, 0.10–0.25 watch")]
    for i, r in enumerate(rows, 3):
        for j, v in enumerate(r, 1):
            ws.cell(row=i, column=j, value=v)
    style_header(ws, 3, 3)
    for r in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        ws.cell(row=r, column=2).number_format = "0.000"
    ws["B6"].number_format = "0.0%"
    widths(ws, [34, 14, 44])

    # ---- Outreach what-if ----
    wo = wb.create_sheet("Outreach")
    wo["A1"] = "Care-team capacity what-if: call the top X% of patients by predicted risk"
    wo["A1"].font = Font(name=FONT, bold=True, size=12, color="1F4E79")
    wo["A2"] = "Capacity (% of discharges)"
    wo["B2"] = 0.20
    wo["B2"].font = INPUT
    wo["B2"].number_format = "0%"
    wo["B2"].fill = PatternFill("solid", start_color="FFFF00")
    wo["B2"].comment = Comment("Edit this cell: pick from 5% to 50% in 5% steps.", "CareSignal")
    order = np.argsort(-p)
    hdr = ("Capacity", "Patients called", "Readmissions reached", "Share of all readmissions", "Precision", "Lift")
    for j, h in enumerate(hdr, 1):
        wo.cell(row=4, column=j, value=h)
    style_header(wo, 4, len(hdr))
    for i, frac in enumerate(np.arange(0.05, 0.501, 0.05)):
        r = 5 + i
        k = int(round(frac * len(p)))
        wo.cell(row=r, column=1, value=round(float(frac), 2)).number_format = "0%"
        wo.cell(row=r, column=2, value=k)
        wo.cell(row=r, column=3, value=int(y[order[:k]].sum()))
        wo.cell(row=r, column=4, value=f"=C{r}/Summary!$B$5").number_format = "0.0%"
        wo.cell(row=r, column=5, value=f"=C{r}/B{r}").number_format = "0.0%"
        wo.cell(row=r, column=6, value=f"=E{r}/Summary!$B$6").number_format = "0.00x"
    last = 5 + i
    wo["A17"] = "At chosen capacity: share of readmissions reached"
    wo["B17"] = f"=IFERROR(INDEX(D5:D{last},MATCH(ROUND(B2,2),A5:A{last},0)),\"pick 5%–50% in 5% steps\")"
    wo["B17"].number_format = "0.0%"
    wo["A18"] = "At chosen capacity: lift over random calling"
    wo["B18"] = f"=IFERROR(INDEX(F5:F{last},MATCH(ROUND(B2,2),A5:A{last},0)),\"\")"
    wo["B18"].number_format = "0.00x"
    ch = LineChart()
    ch.add_data(Reference(wo, min_col=4, min_row=4, max_row=last), titles_from_data=True)
    ch.set_categories(Reference(wo, min_col=1, min_row=5, max_row=last))
    tidy(ch, "Readmissions reached vs outreach capacity", "share of patients called", "share of readmissions")
    wo.add_chart(ch, "H4")
    widths(wo, [42, 16, 20, 22, 12, 10])

    # ---- Calibration ----
    wc = wb.create_sheet("Calibration")
    hdr = ("Risk decile", "Encounters", "Sum of predicted risk", "Readmissions", "Mean predicted", "Observed rate", "Gap")
    for j, h in enumerate(hdr, 1):
        wc.cell(row=1, column=j, value=h)
    style_header(wc, 1, len(hdr))
    for i, idx in enumerate(np.array_split(np.argsort(p), 10)):
        r = 2 + i
        wc.cell(row=r, column=1, value=i + 1)
        wc.cell(row=r, column=2, value=len(idx))
        wc.cell(row=r, column=3, value=round(float(p[idx].sum()), 4))
        wc.cell(row=r, column=4, value=int(y[idx].sum()))
        wc.cell(row=r, column=5, value=f"=C{r}/B{r}").number_format = "0.0%"
        wc.cell(row=r, column=6, value=f"=D{r}/B{r}").number_format = "0.0%"
        wc.cell(row=r, column=7, value=f"=F{r}-E{r}").number_format = "+0.0%;-0.0%;0.0%"
    wc["A13"] = "Weighted mean |gap| (ECE)"
    wc["G13"] = "=SUMPRODUCT(ABS(G2:G11),B2:B11)/SUM(B2:B11)"
    wc["G13"].number_format = "0.000"
    ch = LineChart()
    ch.add_data(Reference(wc, min_col=5, max_col=6, min_row=1, max_row=11), titles_from_data=True)
    ch.set_categories(Reference(wc, min_col=1, min_row=2, max_row=11))
    tidy(ch, "Predicted vs observed readmission by risk decile", "risk decile", "rate")
    wc.add_chart(ch, "I2")
    widths(wc, [12, 12, 20, 14, 15, 14, 10])

    # ---- Fairness ----
    wf = wb.create_sheet("Fairness")
    fa = pd.read_csv("reports/fairness_audit_test.csv")
    fa["attribute"] = fa.attribute.map({"race_audit": "Race (audit only)", "is_male": "Sex", "age_band": "Age band"})
    sex = {"0": "Female", "1": "Male"}
    fa["group"] = fa.apply(lambda r: sex.get(str(r.group), r.group) if r.attribute == "Sex" else r.group, axis=1)
    hdr = ("Attribute", "Group", "Encounters", "Readmissions", "AUROC", "AUROC CI low", "AUROC CI high",
           "Share of readmissions flagged (TPR)", "TPR gap vs overall")
    for j, h in enumerate(hdr, 1):
        wf.cell(row=1, column=j, value=h)
    style_header(wf, 1, len(hdr))
    for i, r in enumerate(fa.itertuples(), 2):
        vals = [r.attribute, str(r.group), int(r.n), int(r.events)] + \
               [None if pd.isna(v) else round(float(v), 4) for v in (r.auroc, r.auroc_lo, r.auroc_hi, r.tpr)]
        for j, v in enumerate(vals, 1):
            wf.cell(row=i, column=j, value=v if v is not None else "n/a (<30 readmissions)" if j == 5 else v)
        for j in (5, 6, 7):
            wf.cell(row=i, column=j).number_format = "0.000"
        wf.cell(row=i, column=8).number_format = "0.0%"
        wf.cell(row=i, column=9, value=f"=IF(ISNUMBER(H{i}),H{i}-$B${len(fa) + 4},\"\")").number_format = "+0.0%;-0.0%;0.0%"
    lr = len(fa) + 4
    thr = res["test"]["flag_threshold_from_valid"]
    wf.cell(row=lr, column=1, value="Overall TPR at the validation threshold")
    wf.cell(row=lr, column=2, value=round(float((p[y == 1] >= thr).mean()), 4)).number_format = "0.0%"
    note = f"Flag threshold = {thr:.3f} (top 20% of validation patients). Race is audited, never a model input."
    wf.cell(row=lr + 1, column=1, value=note)
    ch = BarChart()
    ch.add_data(Reference(wf, min_col=8, min_row=1, max_row=len(fa) + 1), titles_from_data=True)
    ch.set_categories(Reference(wf, min_col=2, min_row=2, max_row=len(fa) + 1))
    tidy(ch, "Share of readmissions flagged, by group", "group", "TPR")
    wf.add_chart(ch, "K2")
    widths(wf, [20, 18, 12, 13, 22, 13, 13, 18, 14])

    # ---- Drift ----
    wd = wb.create_sheet("Drift")
    dr = pd.read_csv("reports/drift_psi_train_vs_test.csv")
    for j, h in enumerate(("Feature", "Type", "PSI (train → test)", "Status"), 1):
        wd.cell(row=1, column=j, value=h)
    style_header(wd, 1, 4)
    for i, r in enumerate(dr.itertuples(), 2):
        wd.cell(row=i, column=1, value=r.feature)
        wd.cell(row=i, column=2, value=r.type)
        wd.cell(row=i, column=3, value=round(float(r.psi), 4)).number_format = "0.000"
        wd.cell(row=i, column=4, value=f'=IF(C{i}<0.1,"stable",IF(C{i}<0.25,"watch","investigate"))')
    n = len(dr) + 1
    wd.cell(row=n + 2, column=1, value="Features to investigate (PSI ≥ 0.25)")
    wd.cell(row=n + 2, column=3, value=f'=COUNTIF(D2:D{n},"investigate")')
    ch = BarChart()
    ch.type = "bar"
    ch.add_data(Reference(wd, min_col=3, min_row=1, max_row=n), titles_from_data=True)
    ch.set_categories(Reference(wd, min_col=1, min_row=2, max_row=n))
    tidy(ch, "Population stability index by feature", "feature", "PSI")
    ch.height = 12
    wd.add_chart(ch, "F2")
    widths(wd, [30, 13, 18, 13])

    # ---- Segments ----
    wsg = wb.create_sheet("Segments")
    sg = pd.read_csv("reports/segments_test.csv")
    hdr = ("Segment", "Encounters", "Readmission rate", "Prior inpatient visits", "ER visits",
           "Days in hospital", "Medications", "Lift vs all test patients")
    for j, h in enumerate(hdr, 1):
        wsg.cell(row=1, column=j, value=h)
    style_header(wsg, 1, len(hdr))
    for i, r in enumerate(sg.itertuples(), 2):
        vals = [f"Segment {chr(65 + i - 2)}", int(r.n), float(r.readmit_30_rate), float(r.number_inpatient),
                float(r.number_emergency), float(r.time_in_hospital), float(r.num_medications)]
        for j, v in enumerate(vals, 1):
            wsg.cell(row=i, column=j, value=v)
        wsg.cell(row=i, column=3).number_format = "0.0%"
        wsg.cell(row=i, column=8, value=f"=C{i}/Summary!$B$6").number_format = "0.00x"
    ch = BarChart()
    ch.add_data(Reference(wsg, min_col=3, min_row=1, max_row=len(sg) + 1), titles_from_data=True)
    ch.set_categories(Reference(wsg, min_col=1, min_row=2, max_row=len(sg) + 1))
    tidy(ch, "30-day readmission rate by K-means segment", "segment", "rate")
    wsg.add_chart(ch, "A8")
    widths(wsg, [12, 12, 16, 20, 10, 16, 13, 22])

    for sheet in wb.worksheets:
        body_font(sheet)
        sheet.freeze_panes = "A2" if sheet.title not in ("Summary", "Outreach") else None
    out = Path("reports/caresignal_dashboard.xlsx")
    wb.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
