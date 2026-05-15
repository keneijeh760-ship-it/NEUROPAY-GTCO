from pathlib import Path
import pandas as pd

EXCEL_PATH = Path("data/verified_sources/nbs/selected food table Mar26.xlsx")

def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Could not find {EXCEL_PATH}")

    excel = pd.ExcelFile(EXCEL_PATH)
    print("Sheets found:")
    for sheet in excel.sheet_names:
        print("-", sheet)

    # Print preview of every sheet so we can see the structure
    for sheet in excel.sheet_names:
        print("\n" + "=" * 80)
        print("SHEET:", sheet)
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=None)
        print(df.head(20).to_string())

if __name__ == "__main__":
    main()