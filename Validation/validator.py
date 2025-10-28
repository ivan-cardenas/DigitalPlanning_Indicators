import sys, json, pathlib
from typing import Dict, Any
import pandas as pd
from jsonschema import validate as js_validate, Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parents[1]

TYPE_MAP = {
    "string": "string",
    "integer": "Int64",
    "number": "float64",
    "boolean": "boolean"
}

def load_csv(path: pathlib.Path, col_spec: Dict[str, Any]) -> pd.DataFrame:
    # Load as strings first, then coerce per column rules for precise errors
    df = pd.read_csv(path, dtype=str).fillna(pd.NA)

    # Ensure required columns
    missing = [c for c in col_spec.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")

    # Coerce types column-by-column with good messages
    for col, spec in col_spec.items():
        t = spec.get("type", "string")
        if t == "string":
            df[col] = df[col].astype("string")
        elif t == "integer":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif t == "number":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t == "boolean":
            df[col] = df[col].str.lower().map({"true": True, "false": False, "1": True, "0": False})
        else:
            raise ValueError(f"{path.name}: unsupported type '{t}' for column '{col}'")

        # Enum
        if "enum" in spec:
            bad = df[~df[col].isin(spec["enum"]) & df[col].notna()]
            if len(bad):
                vals = sorted(set(bad[col].dropna().astype(str)))
                raise ValueError(f"{path.name}: column '{col}' has values not in enum {spec['enum']}: {vals}")

        # min/max (numbers only)
        if t in ("integer", "number"):
            if "min" in spec:
                bad = df[df[col].notna() & (df[col] < spec["min"])]
                if len(bad):
                    raise ValueError(f"{path.name}: column '{col}' has values < {spec['min']}")
            if "max" in spec:
                bad = df[df[col].notna() & (df[col] > spec["max"])]
                if len(bad):
                    raise ValueError(f"{path.name}: column '{col}' has values > {spec['max']}")

    return df

def check_primary_key(df: pd.DataFrame, table: str, pk: str):
    if pk not in df.columns:
        raise ValueError(f"{table}: primary key column '{pk}' not found")
    if df[pk].isna().any():
        raise ValueError(f"{table}: primary key '{pk}' contains nulls")
    dup = df[df[pk].duplicated(keep=False)]
    if len(dup):
        ids = dup[pk].astype(str).tolist()[:10]
        raise ValueError(f"{table}: primary key '{pk}' has duplicates (e.g., {ids}...)")

def check_foreign_keys(df_map: Dict[str, pd.DataFrame], schema: Dict[str, Any]):
    for tname, tcfg in schema["tables"].items():
        fks = tcfg.get("foreign_keys", [])
        for fk in fks:
            col = fk["column"]
            ref_table = fk["ref_table"]
            ref_col = fk["ref_column"]
            if tname not in df_map:
                raise ValueError(f"FK check: table '{tname}' not loaded")
            if ref_table not in df_map:
                raise ValueError(f"FK check: referenced table '{ref_table}' not loaded")
            left = df_map[tname][col].dropna().astype(str).unique()
            right = set(df_map[ref_table][ref_col].dropna().astype(str).unique())
            missing_refs = sorted([v for v in left if v not in right])[:20]
            if missing_refs:
                raise ValueError(
                    f"Foreign key {tname}.{col} -> {ref_table}.{ref_col} "
                    f"has {len(missing_refs)} missing values (e.g., {missing_refs}...)"
                )

def main():
    schema_path = ROOT / "schema.json"
    if not schema_path.exists():
        print("schema.json not found at repo root", file=sys.stderr)
        sys.exit(2)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # Optional: validate the schema itself is well-formed JSON (light check)
    Draft202012Validator.check_schema({
        "type": "object",
        "properties": {
            "tables": {"type": "object"}
        },
        "required": ["tables"]
    })

    df_map = {}
    # Load & validate each table
    for tname, tcfg in schema["tables"].items():
        path = ROOT / tcfg["path"]
        if not path.exists():
            raise ValueError(f"{tname}: file not found at '{tcfg['path']}'")

        columns = tcfg.get("columns", {})
        df = load_csv(path, columns)

        # Primary key
        pk = tcfg.get("primary_key")
        if pk:
            check_primary_key(df, tname, pk)

        df_map[tname] = df

    # Foreign keys across tables
    check_foreign_keys(df_map, schema)

    print("✅ Data validation passed for all tables and relationships.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Validation failed:\n{e}", file=sys.stderr)
        sys.exit(1)
