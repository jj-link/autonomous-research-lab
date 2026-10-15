"""Grade one approved long response without embedding its text in a manifest."""
import json
from pathlib import Path

from grade import grade_job, load_grading_file, write_reports


def main():
    job = json.loads(Path("/experiment/settings.json").read_text())
    job["generation"] = json.loads(Path("/inputs/response/generation.json").read_text())
    if job["generation"]["id"] != job["id"] or job["generation"]["family"] != job["family"]:
        raise ValueError("The response differs from its approved grading identity")
    job["grading"] = load_grading_file(job["grading_file"])
    write_reports(grade_job(job), Path("/workspace"))


if __name__ == "__main__":
    main()
