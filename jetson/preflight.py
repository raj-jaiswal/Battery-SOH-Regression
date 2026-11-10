from pathlib import Path
import argparse, json, sys
import pandas as pd
MODELS = [
    "gpr_rbf","gpr_matern","gpr_ard","sparse_gpr","deep_kernel_gpr",
    "svr","random_forest","xgboost","wide_nn","deep_nn","wide_deep_nn",
    "bilstm","symbolic_regression"
]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=None,help="PBL project root; default: parent of this script")
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve() if args.project else Path(__file__).resolve().parents[1]
    problems=[]
    parquet=root/"data"/"unified.parquet"
    if not parquet.is_file():
        problems.append(f"Missing {parquet}")
    else:
        try:
            df=pd.read_parquet(parquet,engine="pyarrow")
            req={"source_dataset","entity_id","cycle_id","cycle_order"}
            if req-set(df.columns):
                problems.append(f"Parquet missing columns: {sorted(req-set(df.columns))}")
            else:
                df["source_dataset"]=df.source_dataset.astype(str).str.upper()
                df=df[df.source_dataset.isin(["NASA","CALCE"])]
                print(f"Parquet: {len(df):,} cycle rows, {df[['source_dataset','entity_id']].drop_duplicates().shape[0]} cells")
                print("Source counts:",df.groupby("source_dataset").size().to_dict())
        except Exception as e:
            problems.append(f"Cannot read genuine Parquet: {e}")
    report_path=root/"models"/"optimized"/"translation_report.json"
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding="utf-8"))
        status={x.get("model"):x.get("status") for x in report}
        print("Translation report:",sum(v=="ok" for v in status.values()),"OK /",sum(v!="ok" for v in status.values()),"failed")
    else:
        status={}
        print("Translation report not found; checking metadata folders only.")
    for model in MODELS:
        folder=root/"models"/"optimized"/model
        meta_path=folder/"metadata.json"
        if not meta_path.is_file():
            problems.append(f"{model}: missing metadata.json")
            continue
        try:
            meta=json.loads(meta_path.read_text(encoding="utf-8"))
            fmt=meta.get("model_format")
            if model in {"gpr_rbf","gpr_matern","gpr_ard","svr","random_forest","sparse_gpr"}:
                required=["model.npz"]
            elif model=="deep_kernel_gpr":
                required=["gp_params.npz",meta.get("onnx_file","feature_extractor.onnx")]
            elif model in {"wide_nn","deep_nn","wide_deep_nn","bilstm"}:
                required=[meta.get("onnx_file","model.onnx")]
            elif model=="xgboost":
                required=[meta.get("file","model.ubj")]
            else:
                required=[meta.get("expression_file","equation.json"),"equation.c"]
            missing=[x for x in required if not (folder/x).is_file()]
            if missing: problems.append(f"{model}: missing files {missing}")
            elif model in status and status[model] != "ok":
                problems.append(f"{model}: translation report status={status[model]}")
            else:
                print(f"OK: {model} ({fmt})")
        except Exception as e:
            problems.append(f"{model}: invalid metadata: {e}")
    if problems:
        print("\nPREFLIGHT FAILED:")
        for p in problems: print(" -",p)
        return 1
    print("\nPREFLIGHT PASSED. You can run model-specific Jetson scripts.")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
