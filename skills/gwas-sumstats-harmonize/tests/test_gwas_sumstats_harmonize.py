"""Tests for gwas-sumstats-harmonize.

Covers the shared harmonization library (column detection, allele logic,
effect derivation), each standalone stage script, the end-to-end wrapper on the
four-format synthetic demo, and (when snakemake is installed) that the
Snakemake engine and the plain-Python engine produce identical output.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "gwas_sumstats_harmonize.py"
WORKFLOW = SKILL_DIR / "workflow"
STAGE_SCRIPTS = WORKFLOW / "scripts"
EXAMPLES = SKILL_DIR / "examples"
CANONICAL = ["SNP", "CHR", "BP", "EA", "NEA", "EAF", "BETA", "SE", "P", "N"]
DEMO_DATASETS = ["cohort_ssf", "cohort_plink2", "cohort_metal", "cohort_regenie"]


def run_cli(args, **kwargs):
    return subprocess.run([sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, **kwargs)


def run_stage(stage, args):
    return subprocess.run(
        [sys.executable, str(STAGE_SCRIPTS / f"{stage}.py")] + args, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def lib():
    spec = importlib.util.spec_from_file_location(
        "gwas_harmonize_lib_under_test", STAGE_SCRIPTS / "lib" / "harmonize_lib.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def reference() -> dict:
    return {(r["CHR"], r["BP"]): r for r in read_tsv(EXAMPLES / "reference.tsv")}


@pytest.fixture(scope="module")
def demo_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    result = run_cli(["--demo", "--output", str(out), "--engine", "python"])
    assert result.returncode == 0, result.stderr
    return out


# ── Library: column detection ────────────────────────────────────────────────


class TestColumnDetection:
    def test_gwas_catalog_ssf(self, lib):
        header = ["chromosome", "base_pair_location", "effect_allele", "other_allele", "beta",
                  "standard_error", "effect_allele_frequency", "p_value", "variant_id", "n"]
        m = lib.detect_columns(header)["mapping"]
        assert m["CHR"] == "chromosome" and m["BP"] == "base_pair_location"
        assert m["EA"] == "effect_allele" and m["NEA"] == "other_allele"
        assert m["EAF"] == "effect_allele_frequency" and m["SNP"] == "variant_id"

    def test_plink2_trio_uses_a1_as_effect(self, lib):
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "A1", "A1_FREQ", "OBS_CT", "OR", "LOG(OR)_SE", "P"]
        det = lib.detect_columns(header)
        assert det["mapping"]["EA"] == "A1"
        assert det["plink2_other"] == ("REF", "ALT")
        assert "NEA" not in det["mapping"]
        assert det["mapping"]["OR"] == "OR" and det["mapping"]["SE"] == "LOG(OR)_SE"
        assert det["mapping"]["N"] == "OBS_CT"

    def test_regenie(self, lib):
        header = ["CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "N", "BETA", "SE", "LOG10P"]
        m = lib.detect_columns(header)["mapping"]
        assert m["EA"] == "ALLELE1" and m["NEA"] == "ALLELE0"
        assert m["BP"] == "GENPOS" and m["LOG10P"] == "LOG10P" and "P" not in m

    def test_maf_is_never_used_as_eaf(self, lib):
        header = ["SNP", "CHR", "BP", "A1", "A2", "MAF", "BETA", "SE", "P"]
        det = lib.detect_columns(header)
        assert "EAF" not in det["mapping"]
        assert any("MAF" in n for n in det["notes"])

    def test_explicit_mapping_wins(self, lib):
        header = ["rsid", "chr", "pos", "alt", "ref", "b", "s", "pv"]
        m = lib.detect_columns(header, explicit={"BETA": "b", "SE": "s", "P": "pv"})["mapping"]
        assert m["BETA"] == "b" and m["SE"] == "s" and m["P"] == "pv"

    def test_explicit_mapping_to_missing_column_errors(self, lib):
        with pytest.raises(lib.HarmonizeError, match="nope"):
            lib.detect_columns(["a", "b"], explicit={"P": "nope"})

    def test_missing_required_reports_what_is_missing(self, lib):
        with pytest.raises(lib.HarmonizeError, match="effect allele"):
            lib.check_derivable(lib.detect_columns(["CHR", "BP", "BETA", "SE", "P"]))


# ── Library: alleles, chromosomes, statistics ────────────────────────────────


class TestAlleleLogic:
    @pytest.mark.parametrize("a, b, expected", [("A", "T", True), ("C", "G", True), ("A", "G", False), ("AT", "A", False)])
    def test_palindromic(self, lib, a, b, expected):
        assert lib.is_palindromic(a, b) is expected

    @pytest.mark.parametrize(
        "ea, nea, ref, alt, expected",
        [
            ("G", "A", "A", "G", "aligned"),
            ("A", "G", "A", "G", "swapped"),
            ("C", "T", "A", "G", "strand_flipped"),
            ("T", "C", "A", "G", "strand_flipped_swapped"),
            ("A", "C", "A", "G", "mismatch"),
            ("T", "A", "A", "T", "aligned"),
            ("A", "T", "A", "T", "swapped"),
        ],
    )
    def test_reference_alignment_cases(self, lib, ea, nea, ref, alt, expected):
        assert lib.classify_alignment(ea, nea, ref, alt) == expected

    @pytest.mark.parametrize("raw, norm", [("chr1", "1"), ("CHR12", "12"), ("23", "X"), ("x", "X"), ("chrM", "MT"), ("7", "7")])
    def test_normalize_chr(self, lib, raw, norm):
        assert lib.normalize_chr(raw) == norm


class TestStatistics:
    def test_p_from_log10p_survives_underflow(self, lib):
        p = lib.p_from_log10p("349.4")
        assert p.endswith("e-350")
        assert lib.valid_p(p)
        assert float(p) == 0.0  # would have been lost as a float

    def test_p_from_log10p_ordinary(self, lib):
        assert math.isclose(float(lib.p_from_log10p("2")), 0.01, rel_tol=1e-6)

    def test_se_from_beta_and_p(self, lib):
        se = lib.se_from_beta_p(0.1, 0.05)
        assert math.isclose(se, 0.1 / 1.959964, rel_tol=1e-5)

    def test_p_from_beta_se(self, lib):
        assert math.isclose(lib.p_from_beta_se(0.1, 0.1 / 1.959964), 0.05, rel_tol=1e-5)

    @pytest.mark.parametrize("p, ok", [("0.5", True), ("1", True), ("0", False), ("1.5", False), ("NA", False), ("", False), ("-1", False)])
    def test_valid_p(self, lib, p, ok):
        assert lib.valid_p(p) is ok

    def test_lambda_gc_null_is_near_one(self, lib):
        ps = [(i + 0.5) / 2000 for i in range(2000)]
        assert abs(lib.lambda_gc(ps) - 1.0) < 0.01


# ── Stage scripts run standalone ──────────────────────────────────────────────


class TestStageScripts:
    def test_map_columns_metal_parses_markername(self, tmp_path):
        out = tmp_path / "m.tsv"
        r = run_stage("map_columns", ["--input", str(EXAMPLES / "cohort_metal.tbl"), "--out", str(out),
                                      "--summary-json", str(tmp_path / "m.json")])
        assert r.returncode == 0, r.stderr
        rows = read_tsv(out)
        assert rows[0]["CHR"] == "1" and rows[0]["BP"] == "1062256"
        assert rows[0]["EA"] == "a"  # case normalisation happens in derive_effects

    def test_map_columns_plink2_other_allele(self, tmp_path):
        out = tmp_path / "p.tsv"
        r = run_stage("map_columns", ["--input", str(EXAMPLES / "cohort_plink2.glm.logistic"), "--out", str(out),
                                      "--summary-json", str(tmp_path / "p.json")])
        assert r.returncode == 0, r.stderr
        row = read_tsv(out)[0]  # REF=C ALT=A A1=C
        assert row["EA"] == "C" and row["NEA"] == "A"

    def test_map_columns_fails_cleanly_on_unusable_file(self, tmp_path):
        bad = tmp_path / "bad.tsv"
        bad.write_text("foo\tbar\n1\t2\n")
        r = run_stage("map_columns", ["--input", str(bad), "--out", str(tmp_path / "o.tsv"),
                                      "--summary-json", str(tmp_path / "o.json")])
        assert r.returncode != 0
        assert "foo" in r.stderr  # shows the header it saw

    def test_derive_effects_from_or(self, tmp_path):
        src = tmp_path / "in.tsv"
        src.write_text("SNP\tCHR\tBP\tEA\tNEA\tEAF\tBETA\tSE\tP\tN\tOR\n"
                       "x\tchr2\t100\tg\ta\t0.3\t\t\t0.05\t100\t2.0\n")
        out = tmp_path / "out.tsv"
        r = run_stage("derive_effects", ["--input", str(src), "--out", str(out), "--summary-json", str(tmp_path / "s.json")])
        assert r.returncode == 0, r.stderr
        row = read_tsv(out)[0]
        assert list(row) == CANONICAL
        assert row["CHR"] == "2" and row["EA"] == "G" and row["NEA"] == "A"
        assert math.isclose(float(row["BETA"]), math.log(2.0), rel_tol=1e-5)
        assert math.isclose(float(row["SE"]), math.log(2.0) / 1.959964, rel_tol=1e-4)
        s = json.loads((tmp_path / "s.json").read_text())
        assert s["derived"]["BETA_from_OR"] == 1 and s["derived"]["SE_from_P"] == 1

    def test_derive_effects_counts_unparseable_log10p(self, tmp_path):
        src = tmp_path / "in.tsv"
        src.write_text("SNP\tCHR\tBP\tEA\tNEA\tEAF\tBETA\tSE\tP\tN\tLOG10P\n"
                       "x\t1\t100\tG\tA\t0.3\t0.1\t\t\t100\tabc\n"
                       "y\t1\t200\tG\tA\t0.3\t0.1\t\t\t100\tinf\n")
        out = tmp_path / "out.tsv"
        r = run_stage("derive_effects", ["--input", str(src), "--out", str(out), "--summary-json", str(tmp_path / "s.json")])
        assert r.returncode == 0, r.stderr
        assert [row["P"] for row in read_tsv(out)] == ["", ""]
        s = json.loads((tmp_path / "s.json").read_text())
        assert s["derived"]["LOG10P_unparseable"] == 2

    def test_qc_filter_reasons(self, tmp_path):
        src = tmp_path / "in.tsv"
        rows = [
            ["ok1", "1", "100", "G", "A", "0.3", "0.1", "0.02", "1e-5", "10"],
            ["badp", "1", "200", "G", "A", "0.3", "0.1", "0.02", "1.5", "10"],
            ["palin", "1", "300", "A", "T", "0.5", "0.1", "0.02", "1e-5", "10"],
            ["badallele", "1", "400", "N", "A", "0.3", "0.1", "0.02", "1e-5", "10"],
            ["nose", "1", "500", "G", "A", "0.3", "0.1", "", "1e-5", "10"],
            ["dup_hi", "1", "100", "A", "G", "0.7", "-0.1", "0.02", "1e-3", "10"],
            ["ok2", "chr1", "50", "C", "T", "0.2", "0.1", "0.02", "0.5", "10"],
        ]
        src.write_text("\t".join(CANONICAL) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
        out = tmp_path / "out.tsv"
        r = run_stage("qc_filter", ["--input", str(src), "--out", str(out), "--summary-json", str(tmp_path / "s.json"),
                                    "--palindromic", "ambiguous", "--min-maf", "0", "--keep-indels", "true"])
        assert r.returncode == 0, r.stderr
        kept = [row["SNP"] for row in read_tsv(out)]
        assert kept == ["ok2", "ok1"]  # sorted by CHR, BP; dup with higher P dropped
        drops = json.loads((tmp_path / "s.json").read_text())["dropped"]
        assert drops == {"invalid_p": 1, "palindromic": 1, "invalid_allele": 1,
                         "missing_required": 1, "duplicate": 1}

    def test_align_reference_without_reference_is_passthrough(self, tmp_path):
        src = EXAMPLES / "reference.tsv"  # any canonical-ish file; just checks passthrough
        canon = tmp_path / "c.tsv"
        canon.write_text("\t".join(CANONICAL) + "\n" + "\t".join(["x", "1", "1", "G", "A", "", "0.1", "0.02", "0.5", ""]) + "\n")
        out = tmp_path / "o.tsv"
        r = run_stage("align_reference", ["--input", str(canon), "--out", str(out),
                                          "--summary-json", str(tmp_path / "s.json"), "--drop-unmatched", "false"])
        assert r.returncode == 0, r.stderr
        assert out.read_text() == canon.read_text()
        assert src.exists()


# ── End to end: demo ──────────────────────────────────────────────────────────


class TestDemoEndToEnd:
    def test_outputs_exist(self, demo_out):
        for name in DEMO_DATASETS:
            assert (demo_out / "harmonized" / f"{name}.tsv").exists(), name
        assert (demo_out / "report.md").exists()
        assert (demo_out / "tables" / "harmonize_summary.tsv").exists()

    def test_written_config_follows_layout(self, demo_out):
        for f in ("data.yaml", "analysis.yaml", "software.yaml"):
            assert (demo_out / "config" / f).exists()
        data = yaml.safe_load((demo_out / "config" / "data.yaml").read_text())
        assert sorted(data["datasets"]) == sorted(DEMO_DATASETS)

    def test_done_sentinels(self, demo_out):
        for stage in ("map_columns", "derive_effects", "qc_filter", "align_reference"):
            for name in DEMO_DATASETS:
                assert (demo_out / "work" / "done" / f"{stage}_{name}.done").exists()

    def test_canonical_columns_and_alleles_match_reference(self, demo_out):
        ref = reference()
        for name in DEMO_DATASETS:
            rows = read_tsv(demo_out / "harmonized" / f"{name}.tsv")
            assert list(rows[0]) == CANONICAL
            for row in rows:
                r = ref[(row["CHR"], row["BP"])]
                assert (row["EA"], row["NEA"]) == (r["ALT"], r["REF"]), (name, row)

    def test_effects_agree_across_formats(self, demo_out):
        """All four formats encode the same truth, so aligned betas must agree."""
        by = {n: {(r["CHR"], r["BP"]): r for r in read_tsv(demo_out / "harmonized" / f"{n}.tsv")} for n in DEMO_DATASETS}
        common = set.intersection(*(set(v) for v in by.values()))
        assert len(common) > 150
        for key in common:
            betas = [float(by[n][key]["BETA"]) for n in DEMO_DATASETS if not (n == "cohort_regenie" and key == min(common))]
            assert max(betas) - min(betas) < 1e-3, (key, betas)

    def test_missing_eaf_filled_from_reference_allele_matched(self, demo_out):
        ref = reference()
        rows = read_tsv(demo_out / "harmonized" / "cohort_ssf.tsv")
        assert all(row["EAF"] != "" for row in rows)
        for row in rows:
            assert math.isclose(float(row["EAF"]), float(ref[(row["CHR"], row["BP"])]["AF"]), abs_tol=1e-3)

    def test_tiny_pvalue_preserved(self, demo_out):
        rows = read_tsv(demo_out / "harmonized" / "cohort_regenie.tsv")
        assert any(row["P"].endswith("e-350") for row in rows)

    def test_result_json(self, demo_out):
        data = json.loads((demo_out / "result.json").read_text())
        assert data["skill"] == "gwas-sumstats-harmonize"
        assert data["engine"] == "python"
        ssf = data["datasets"]["cohort_ssf"]
        assert ssf["rows_in"] == 201
        assert ssf["dropped"]["invalid_p"] == 2
        assert ssf["dropped"]["duplicate"] == 1
        assert ssf["alignment"]["swapped"] > 0 and ssf["alignment"]["strand_flipped"] > 0
        assert 0.5 < ssf["lambda_gc"] < 2.0

    def test_report(self, demo_out):
        text = (demo_out / "report.md").read_text(encoding="utf-8")
        assert "not a medical device" in text
        for name in DEMO_DATASETS:
            assert name in text
        assert "λGC" in text or "lambda" in text.lower()

    def test_reproducibility_bundle(self, demo_out):
        repro = demo_out / "reproducibility"
        for f in ("commands.sh", "environment.yml", "checksums.sha256"):
            assert (repro / f).exists()
        cmds = (repro / "commands.sh").read_text()
        assert "map_columns.py" in cmds and "align_reference.py" in cmds


class TestWrapperInputs:
    def test_single_file_input_with_column_override(self, tmp_path):
        src = tmp_path / "weird.txt"
        src.write_text("marker,c,pos,eff,oth,b,s,pv\n"
                       "rs1,1,100,G,A,0.1,0.02,1e-5\n"
                       "rs2,1,200,C,T,-0.1,0.02,0.3\n")
        cols = tmp_path / "cols.yaml"
        cols.write_text(yaml.safe_dump({"SNP": "marker", "CHR": "c", "BP": "pos", "EA": "eff",
                                        "NEA": "oth", "BETA": "b", "SE": "s", "P": "pv"}))
        out = tmp_path / "out"
        r = run_cli(["--input", str(src), "--columns", str(cols), "--output", str(out), "--engine", "python"])
        assert r.returncode == 0, r.stderr
        rows = read_tsv(out / "harmonized" / "weird.tsv")
        assert [r["SNP"] for r in rows] == ["rs1", "rs2"]

    def test_unusable_input_gives_clear_error(self, tmp_path):
        src = tmp_path / "bad.tsv"
        src.write_text("foo\tbar\n1\t2\n")
        r = run_cli(["--input", str(src), "--output", str(tmp_path / "o"), "--engine", "python"])
        assert r.returncode != 0
        assert "map_columns" in r.stderr

    def test_no_input_errors(self, tmp_path):
        assert run_cli(["--output", str(tmp_path)]).returncode != 0


# ── Output contract ───────────────────────────────────────────────────────────


def _parse_output_contract(skill_md: Path) -> list[str]:
    text = skill_md.read_text(encoding="utf-8")
    m = re.search(r"##\s*Output Structure\s*\n+```[^\n]*\n(.*?)\n```", text, re.S)
    if not m:
        return []
    files, parents = [], {}
    for raw in m.group(1).splitlines():
        if not raw.strip():
            continue
        parts = re.split(r"\s+#", raw, maxsplit=1)
        entry, comment = parts[0], (parts[1] if len(parts) > 1 else "")
        prefix, name = re.match(r"^([\s│├└─]*)(.*)$", entry).groups()
        name = name.strip()
        depth = len(prefix) // 4
        if not name or depth == 0:
            continue
        if name.endswith("/"):
            parents[depth] = name.rstrip("/")
            for d in [k for k in parents if k > depth]:
                del parents[d]
            continue
        if "optional" in comment.lower():
            continue
        rel = "/".join(parents[d] for d in sorted(parents) if d < depth)
        files.append(f"{rel}/{name}" if rel else name)
    return files


class TestOutputContract:
    def test_documented_outputs_are_produced(self, demo_out):
        promised = _parse_output_contract(SKILL_DIR / "SKILL.md")
        assert promised, "SKILL.md must document an Output Structure tree"
        missing = [p for p in promised if not (demo_out / p).exists()]
        assert not missing, "SKILL.md promises artifacts the skill did not produce: " + ", ".join(missing)


# ── Snakemake engine ──────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("snakemake") is None, reason="snakemake not installed")
class TestSnakemakeEngine:
    def test_snakemake_engine_matches_python_engine(self, demo_out, tmp_path):
        out = tmp_path / "sm"
        r = run_cli(["--demo", "--output", str(out), "--engine", "snakemake"])
        assert r.returncode == 0, r.stderr
        assert json.loads((out / "result.json").read_text())["engine"] == "snakemake"
        for name in DEMO_DATASETS:
            a = (demo_out / "harmonized" / f"{name}.tsv").read_text()
            b = (out / "harmonized" / f"{name}.tsv").read_text()
            assert a == b, name

    def test_workflow_dry_run_against_written_config(self, demo_out, tmp_path):
        work = tmp_path / "copy"
        shutil.copytree(demo_out, work)
        r = subprocess.run(
            [sys.executable, "-m", "snakemake", "-n", "--cores", "1",
             "-s", str(WORKFLOW / "Snakefile"), "-d", str(work),
             "--config", f"config_dir={(work / 'config').as_posix()}"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
