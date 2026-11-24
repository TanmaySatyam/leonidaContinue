"""Semantic tests for RDNA2 SGPR tuples spanning past s105 into vcc_lo (106) / vcc_hi (107).

Xbox RDNA2 shaders emit 4- and 8-dword SGPR operands such as s[104:107] and s[100:107]
(LLVM VCCTuple syntax `[s104,s105,vcc_lo,vcc_hi]` and `[s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi]`)
when VCC is dead. Both spellings must normalize to the same `slist` operand in shader_ir and translate
cleanly through tools/rdna2spirv.py with s106/s107 aliasing vcc_lo/vcc_hi.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
from shader_ir import operand, parse_line  # noqa: E402


def _assemble_and_validate(spvasm: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "s.spvasm")
        spv_path = os.path.join(tmp, "s.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        p1 = subprocess.run([spirv_as, "--target-env", "vulkan1.3", asm_path, "-o", spv_path],
                            capture_output=True, text=True)
        if p1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{p1.stderr}")
        p2 = subprocess.run([spirv_val, "--target-env", "vulkan1.3", spv_path],
                            capture_output=True, text=True)
        if p2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{p2.stderr}")


def _translate_cs(lines: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        disasm_dir = os.path.join(tmp, "disasm")
        fspx_dir = os.path.join(tmp, "fspx")
        os.makedirs(disasm_dir)
        os.makedirs(fspx_dir)
        with open(os.path.join(fspx_dir, "cs_sgpr_high_range.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({
                "threads": [64, 1, 1],
                "wave32": True,
                "user_sgprs": 8,
                "tgid_enable": [1, 0, 0],
                "tidig_comp_cnt": 0,
                "lds_bytes": 0,
                "rsrc2": 0x90,
            }, f)
        s_path = os.path.join(disasm_dir, "cs_sgpr_high_range.s")
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("; synthetic compute shader test\n")
            for line in lines:
                f.write(line + "\n")
        res = rdna2spirv.translate(s_path)
        return res[0] if isinstance(res, tuple) else res


class SgprHighRangeTest(unittest.TestCase):
    def test_shader_ir_normalizes_high_sgpr_ranges_to_vcc_tuples(self):
        self.assertEqual(
            operand("s[104:107]"),
            {"k": "slist", "r": [104, 105, "vcc_lo", "vcc_hi"], "n": 4},
        )
        self.assertEqual(
            operand("s[100:107]"),
            {"k": "slist", "r": [100, 101, 102, 103, 104, 105, "vcc_lo", "vcc_hi"], "n": 8},
        )
        self.assertEqual(
            operand("s[105:106]"),
            {"k": "slist", "r": [105, "vcc_lo"], "n": 2},
        )
        # Ranges ending at or below s105 remain ordinary sgpr ranges.
        self.assertEqual(
            operand("s[102:105]"),
            {"k": "sgpr", "r": 102, "n": 4, "neg": False, "abs": False},
        )
        # Bracketed VCCTuple syntax produces the exact same structure.
        self.assertEqual(
            operand("s[104:107]"),
            operand("[s104,s105,vcc_lo,vcc_hi]"),
        )
        self.assertEqual(
            operand("s[100:107]"),
            operand("[s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi]"),
        )

    def test_smem_and_mubuf_high_sgpr_ranges_translate_and_alias_vcc(self):
        for r4, r8 in (
            ("s[104:107]", "s[100:107]"),
            ("[s104,s105,vcc_lo,vcc_hi]", "[s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi]"),
        ):
            with self.subTest(r4=r4, r8=r8):
                spvasm = _translate_cs([
                    f"\ts_load_dwordx4 {r4}, s[0:1], 0x0",
                    f"\ts_load_dwordx8 {r8}, s[0:1], 0x20",
                    "\ts_waitcnt lgkmcnt(0)",
                    f"\ts_buffer_load_dword s10, {r4}, 0x0",
                    "\ts_waitcnt lgkmcnt(0)",
                    "\tv_mov_b32_e32 v1, vcc_lo",
                    "\tv_add_nc_u32_e32 v1, vcc_hi, v1",
                    "\tv_add_nc_u32_e32 v1, s10, v1",
                    f"\tbuffer_store_dword v1, v0, {r4}, 0 offen",
                    "\ts_endpgm",
                ])
                # s106/s107 writes from s_load_dwordx4/x8 must target %vcc / %vcc_hi.
                self.assertIn("%vcc", spvasm)
                self.assertIn("%vcc_hi", spvasm)
                _assemble_and_validate(spvasm)

    def test_mimg_high_sgpr_rsrc_and_sampler_translate(self):
        for r8, samp4 in (
            ("s[100:107]", "s[104:107]"),
            ("[s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi]", "[s104,s105,vcc_lo,vcc_hi]"),
        ):
            with self.subTest(r8=r8, samp4=samp4):
                spvasm = _translate_cs([
                    f"\ts_load_dwordx8 {r8}, s[0:1], 0x0",
                    "\ts_load_dwordx4 s[8:11], s[0:1], 0x40",
                    "\ts_waitcnt lgkmcnt(0)",
                    "\tv_mov_b32_e32 v2, 0.5",
                    "\tv_mov_b32_e32 v3, 0.5",
                    f"\timage_sample_lz v[4:7], v[2:3], {r8}, s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D",
                    "\ts_waitcnt vmcnt(0)",
                    f"\ts_load_dwordx4 {samp4}, s[0:1], 0x80",
                    "\ts_waitcnt lgkmcnt(0)",
                    f"\timage_gather4_lz v[8:11], v[2:3], s[16:23], {samp4} dmask:0x1 dim:SQ_RSRC_IMG_2D",
                    "\ts_waitcnt vmcnt(0)",
                    "\tbuffer_store_dword v4, v0, s[8:11], 0 offen",
                    "\tbuffer_store_dword v8, v0, s[8:11], 0 offen offset:4",
                    "\ts_endpgm",
                ])
                self.assertIn("OpImage", spvasm)
                _assemble_and_validate(spvasm)


if __name__ == "__main__":
    unittest.main()
