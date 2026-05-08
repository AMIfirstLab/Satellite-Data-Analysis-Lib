from pathlib import Path

from satprep.export.report import export_quality_reports_to_csv, export_quality_reports_to_json, summarize_quality_reports
from satprep.quality.report import analyze_chip_file


def main() -> None:
    # 칩 폴더의 GeoTIFF 파일을 순회하며 품질 점수를 계산한다.
    reports = [analyze_chip_file(path) for path in sorted(Path("chips").glob("*.tif"))]
    export_quality_reports_to_json(reports, "reports/quality.json")
    export_quality_reports_to_csv(reports, "reports/quality.csv")
    print(summarize_quality_reports(reports))


if __name__ == "__main__":
    main()

