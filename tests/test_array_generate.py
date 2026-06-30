"""Tests for rosi.array.generate — CSV array generation."""

import csv
import math

from rosi.array.generate import generate_cylindrical_csv


class TestGenerateCylindricalCsv:
    def test_row_count(self, tmp_path):
        out = tmp_path / "test.csv"
        generate_cylindrical_csv(str(out), 10, 2.0, 1.5)
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 11  # header + 10 data rows

    def test_header(self, tmp_path):
        out = tmp_path / "test.csv"
        generate_cylindrical_csv(str(out), 6, 1.0, 0.5)
        with open(out) as f:
            header = f.readline().strip()
        assert header == "x,y,z"

    def test_radius(self, tmp_path):
        out = tmp_path / "test.csv"
        generate_cylindrical_csv(str(out), 12, 3.0, 1.0)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                x, y = float(row[0]), float(row[1])
                r = math.sqrt(x**2 + y**2)
                assert abs(r - 3.0) < 1e-5

    def test_z_constant(self, tmp_path):
        out = tmp_path / "test.csv"
        generate_cylindrical_csv(str(out), 8, 1.5, 2.5)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                assert float(row[2]) == 2.5

    def test_parsable_floats(self, tmp_path):
        out = tmp_path / "test.csv"
        generate_cylindrical_csv(str(out), 5, 1.0, 1.0)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                [float(v) for v in row]  # should not raise
