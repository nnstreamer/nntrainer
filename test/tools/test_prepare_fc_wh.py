#!/usr/bin/env python3
# @brief Offline WH preparation and the actual portable C++ runtime reader.
# SPDX-License-Identifier: Apache-2.0
"""Offline WH preparation and the actual portable C++ runtime reader.

Run: python3 -m unittest discover -s test/tools -p test_prepare_fc_wh.py -v
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare_fc_wh", ROOT / "tools/htp/prepare_fc_wh.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def source_weight(k, n):
    # All +1 with one -8 value in an off-diagonal tile. Build the nibble
    # location independently from the documented byte layout, not the tool.
    packed = bytearray(b"\x11" * (k * n // 2))
    row, col = 39, n - 17
    tile = ((row // 32) * (n // 32) + col // 32) * 512
    byte = tile + ((row % 32) // 8) * 128 + (col % 32) * 4 + row % 4
    shift = ((row // 4) % 2) * 4
    packed[byte] = (packed[byte] & ~(15 << shift)) | (8 << shift)
    scales = [0.25 + (j % 7) * 0.125 for j in range(n)]
    sums = [k] * n
    sums[col] -= 9
    return bytes(packed) + struct.pack(f"<{n}f", *scales) + struct.pack(f"<{n}f", *sums)


def unpack_value(packed, k, n, row, column):
    tile = ((row // 32) * (n // 32) + column // 32) * 512
    byte = tile + ((row % 32) // 8) * 128 + (column % 32) * 4 + row % 4
    value = (packed[byte] >> (((row // 4) % 2) * 4)) & 15
    return value - 16 if value >= 8 else value


class PrepareFcWhTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("a C++ compiler is required to test the runtime reader")
        driver = Path(cls.build.name) / "reader.cpp"
        cls.reader = Path(cls.build.name) / "reader"
        driver.write_text(r'''
#include <htp_prepared_fc.h>
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char **argv) {
  if (argc != 6) return 2;
  try {
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(input)), {});
    const uint64_t off = std::stoull(argv[3]);
    const uint32_t K = std::stoul(argv[4]), N = std::stoul(argv[5]);
    const size_t wh = size_t(K) * N / 2;
    if (off + wh + size_t(N) * 8 > data.size()) return 2;
    // Copy the floats into an aligned vector instead of aliasing byte storage.
    std::vector<float> scales(2 * N);
    std::memcpy(scales.data(), data.data() + off + wh, N * 8);
    const auto hash = nntrainer::preparedFcSourceHash(data.data() + off, scales.data(), K, N);
    const auto prepared = nntrainer::readPreparedFc(argv[1], off, K, N, hash);
    std::cout << prepared.slices.size() << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
''')
        subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT / "nntrainer/tensor/htp_backend"),
                        str(driver), "-o", str(cls.reader)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.model = self.directory / "model.bin"
        self.manifest = self.directory / "manifest.json"
        self.output = self.directory / "prepared"
        self.weight = {"file_offset": 128, "K": 64, "N": 96, "name": "conv.in_proj"}
        self.source = source_weight(64, 96)
        self.model.write_bytes(b"\0" * 128 + self.source + b"\0" * 256)
        self.document = {"schema_version": 1, "dtype": "QS4CX_WH",
                         "model_size": self.model.stat().st_size, "weights": [self.weight]}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.document))

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            PREP.prepare(self.model, self.manifest, self.output)
        return self.output / PREP.weight_name(self.weight)

    def cpp_accepts(self, path):
        result = subprocess.run([str(self.reader), str(path), str(self.model),
                                 str(self.weight["file_offset"]), str(self.weight["K"]),
                                 str(self.weight["N"])], capture_output=True, text=True)
        return result.returncode == 0

    def rejects(self, path):
        with self.assertRaises(ValueError):
            PREP.verify_weight(path, PREP.fnv1a64(self.source), self.weight)
        self.assertFalse(self.cpp_accepts(path))

    def test_roundtrip_matches_native_runtime_and_integer_metadata(self):
        path = self.prepare()
        self.assertTrue(self.cpp_accepts(path))
        self.assertEqual(PREP.prepare(self.model, self.manifest, self.output, True), 1)
        data = path.read_bytes()
        start, cols, offset = PREP.DESCRIPTOR.unpack_from(data, PREP.HEADER.size)
        self.assertEqual((start, cols, offset), (0, 96, 512))
        wh = data[PREP.HEADER.size + offset:]
        self.assertEqual(wh[:64 * 96 // 2], self.source[:64 * 96 // 2])
        sums = struct.unpack_from("<96i", wh, 64 * 96 // 2 + 4 * 96)
        self.assertEqual(sums[79], 55)
        self.assertEqual(sums[0], 64)

    def test_multislice_copies_tiles_scales_and_tail_without_requantization(self):
        k, n = 2048, 2080
        weight = {"file_offset": 128, "K": k, "N": n}
        source = source_weight(k, n)
        payload, count = PREP.make_payload(source, k, n)
        self.assertEqual(count, 2)
        for index, (start, columns) in enumerate(((0, 2048), (2048, 32))):
            col0, width, off = PREP.DESCRIPTOR.unpack_from(payload, index * 16)
            self.assertEqual((col0, width), (start, columns))
            self.assertEqual(off % 512, 0)
            part = payload[off:off + k * columns // 2]
            for row in (0, 4, 7, 8, 31, 32, 39, k - 1):
                for column in (0, 15, columns - 17, columns - 1):
                    self.assertEqual(unpack_value(part, k, columns, row, column),
                                     unpack_value(source, k, n, row, start + column))
            scales_start = off + k * columns // 2
            self.assertEqual(payload[scales_start:scales_start + 4 * columns],
                             source[k * n // 2 + 4 * start:k * n // 2 + 4 * (start + columns)])
        # Also exercise the actual parser's multi-descriptor validation.
        self.weight, self.source = weight, source
        self.model.write_bytes(b"\0" * 128 + source)
        path = self.directory / "multi.nntrwh"
        path.write_bytes(PREP.encode_weight(source, PREP.fnv1a64(source), weight))
        self.assertTrue(self.cpp_accepts(path))

    def test_payload_and_padding_corruption_rejected(self):
        path = self.prepare()
        original = path.read_bytes()
        for at in (PREP.HEADER.size + 32, len(original) - 1):
            with self.subTest(at=at):
                data = bytearray(original)
                data[at] ^= 1
                path.write_bytes(data)
                self.rejects(path)

    def test_stale_source_rejected_even_with_same_size_and_shape(self):
        path = self.prepare()
        data = bytearray(self.model.read_bytes())
        data[self.weight["file_offset"] + 123] ^= 1
        self.model.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "identity"):
            PREP.prepare(self.model, self.manifest, self.output, True)
        self.assertFalse(self.cpp_accepts(path))

    def test_truncated_and_oversized_header_rejected(self):
        path = self.prepare()
        original = path.read_bytes()
        for size in (12, 64, len(original) - 1):
            path.write_bytes(original[:size])
            self.rejects(path)
        data = bytearray(original)
        struct.pack_into("<Q", data, 48, 1 << 40)
        path.write_bytes(data)
        self.rejects(path)

    def test_wrong_shape_and_slice_bounds_rejected_with_valid_payload_hash(self):
        path = self.prepare()
        original = path.read_bytes()
        data = bytearray(original)
        struct.pack_into("<I", data, 40, 32)
        path.write_bytes(data)
        self.rejects(path)
        data = bytearray(original)
        struct.pack_into("<Q", data, 64 + 8, 1 << 40)
        struct.pack_into("<Q", data, 56, PREP.fnv1a64(data[64:]))
        path.write_bytes(data)
        self.rejects(path)

    def test_manifest_bounds_shapes_size_and_overlap(self):
        for change in ({"file_offset": self.model.stat().st_size}, {"K": 33},
                       {"N": 0}, {"K": True}):
            with self.subTest(change=change):
                self.document["weights"] = [dict(self.weight, **change)]
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())
        self.document["weights"] = [self.weight, dict(self.weight)]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.prepare()
        self.document["weights"] = [self.weight]
        self.document["model_size"] -= 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "model_size"):
            self.prepare()

    def test_invalid_metadata_and_pinned_hash_do_not_publish_partial_output(self):
        for invalid in (float("nan"), float("inf"), 0.5, 100000.0):
            data = bytearray(self.source)
            struct.pack_into("<f", data, 64 * 96 // 2 + 4 * 96, invalid)
            self.model.write_bytes(b"\0" * 128 + data + b"\0" * 256)
            with self.assertRaisesRegex(ValueError, "column sums"):
                self.prepare()
            self.assertFalse(self.output.exists())
        self.model.write_bytes(b"\0" * 128 + self.source + b"\0" * 256)
        self.weight["source_fnv1a64"] = "0000000000000000"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "source_fnv1a64"):
            self.prepare()
        self.assertFalse(self.output.exists())
        # The first artifact has already been generated when a later pinned
        # hash fails; no incomplete output directory may become visible.
        self.weight.pop("source_fnv1a64")
        second = dict(self.weight, file_offset=128 + len(self.source),
                      source_fnv1a64="0000000000000000")
        self.model.write_bytes(b"\0" * 128 + self.source + self.source)
        self.document["model_size"] = self.model.stat().st_size
        self.document["weights"] = [self.weight, second]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "source_fnv1a64"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.directory.glob(".nntrwh-*")))

    def test_existing_output_is_not_overwritten(self):
        path = self.prepare()
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
