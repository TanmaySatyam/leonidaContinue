"""Disassemble the native RDNA2 code in unpacked FSSM shader blobs with LLVM's AMDGPU disassembler.

FSSM header (little-endian qwords at 0x00..): 'FSSM' 01 01, blob size, ..., then a table where the
code range appears as (offset, size) = qword[6], qword[7] in every blob checked so far (e.g. 0x2f8, 0x100).
The code ends with s_endpgm (0xBF810000). If the header guess doesn't hold for a blob, the tool falls back to
scanning back from the first s_endpgm to the start that decodes with no invalid instructions.

Usage:
    python -I shader_disasm.py <fssm dir or file> [--unpack-index FILE] [--out DIR] [--mcpu gfx1013] [--limit N] [--skip-existing]
Writes <out>/<name>.s and prints a summary (decoded cleanly / with invalid words / failed).
Set LLVM_MC to llvm-mc's path, or to a whole command such as
'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc' when the tool is not on PATH. Off Windows, a versioned install
(llvm-mc-21, /usr/lib/llvm-21/bin/llvm-mc) is found without it (tools/mc_command.py).
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mc_command import install_hint, mc_command, not_found  # noqa: E402
from report_paths import path_key, same_report_path  # noqa: E402
from shader_stage_manifest import shader_stems  # noqa: E402
import fssm_header  # noqa: E402
import rdna2_decode  # noqa: E402

TRIPLE = "amdgcn--amdpal"
ENDPGM = b"\x00\x00\x81\xbf"
PROBE = "0x00 0x00 0x81 0xbf"  # s_endpgm: the one instruction every shader ends with
FAILED = []                    # (path, first stderr line) for shaders that produced no disassembly


MC = mc_command()


def fail(msg: str) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def output_path(path: str, out: str) -> str:
    return os.path.join(out, os.path.splitext(os.path.basename(path))[0] + ".s")


def validate_output(out: str, files: list) -> None:
    """Refuse a planned <out>/<stem>.s that is a folder or names an input, before probing llvm-mc or creating --out.
    A single `foo.s` given with --out at its own folder (or the default ../disasm of a file already in disasm/) was
    replaced by its own listing; on Windows and macOS `FOO.S` is the same file as the `FOO.s` written for it."""
    inputs = {path_key(p): p for p in files}
    for src in files:
        dst = output_path(src, out)
        if os.path.isdir(dst):
            fail(f"cannot write {dst}: destination is a folder, not a file")
        hit = inputs.get(path_key(dst))
        if hit is None and same_report_path(src, dst):  # a hard link or another spelling the name rules missed
            hit = src
        if hit is not None:
            fail(f"output {dst} would overwrite input {hit}")


def positive_int(text: str) -> int:
    """--jobs 0 made ThreadPoolExecutor raise, and --limit -1 silently dropped the last file."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def probe(mcpu: str) -> None:
    """Disassemble one s_endpgm first, so a missing tool or AMDGPU target fails now and not per shader."""
    try:
        p = subprocess.run(MC + ["--disassemble", f"-triple={TRIPLE}", f"-mcpu={mcpu}"],
                           input=PROBE, capture_output=True, text=True, errors="replace",
                           creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    except OSError as e:
        fail(not_found(MC[0], e))
    if p.returncode != 0 or "s_endpgm" not in p.stdout:
        why = (p.stderr.strip().splitlines() or [f"exit code {p.returncode}"])[0]
        fail(f"'{' '.join(MC)}' did not disassemble an s_endpgm: {why}\n"
             f"The disassembler needs LLVM 21+ with the AMDGPU target (docs/design.md, 'Disassembling shaders').\n"
             f"{install_hint()}")


ENCODING = re.compile(r"; encoding: \[([^\]]*)\]")
INVALID_AT = re.compile(r":1:(\d+): warning: invalid instruction encoding")

_SMEM_DATA_WIDTH = {
    0x00: 1, 0x01: 2, 0x02: 4, 0x03: 8, 0x04: 16,
    0x08: 1, 0x09: 2, 0x0A: 4, 0x0B: 8, 0x0C: 16,
    0x10: 1, 0x11: 2, 0x12: 4,
    0x18: 1, 0x19: 2, 0x1A: 4,
}


def _vcc_tuple_str(start: int, width: int) -> str:
    """Format an SGPR range spanning past s105 into vcc_lo (106) / vcc_hi (107) in LLVM VCCTuple syntax."""
    items = [f"s{r}" if r <= 105 else ("vcc_lo" if r == 106 else "vcc_hi") for r in range(start, start + width)]
    return "[" + ",".join(items) + "]"


def _pick_placeholder(width: int, *avoid: tuple[int, int]) -> int:
    """Choose a low aligned SGPR base (0..96) of `width` dwords that does not overlap any `(base, w)` in `avoid`."""
    align = max(4, width) if width in (4, 8, 16) else 2
    for base in range(0, 96, align):
        if all(base + width <= b or base >= b + w for b, w in avoid):
            return base
    return 0


def _sgpr_vcc_patches(code: bytes, offsets: set) -> dict:
    """Identify rejected 64-bit SMEM/MIMG/MUBUF instructions at `offsets` whose SGPR tuple spans into vcc_lo (106)
    or vcc_hi (107). llvm-mc rejects SGPR_128/256/512 tuples crossing s105, even though Xbox RDNA2 shaders emit
    them when VCC is dead. Returns {offset: (orig_w0, orig_w1, patch_w0, patch_w1, replacements)} where each
    replacement is (placeholder_text, restored_vcc_tuple_text)."""
    patches = {}
    for at in sorted(offsets):
        if at % 4 or at + 8 > len(code):
            continue
        w0, w1 = struct.unpack_from("<2I", code, at)
        hi6 = w0 >> 26
        pw0, pw1 = w0, w1
        reps = []
        if hi6 == 0x3D:  # SMEM (64-bit)
            op = (w0 >> 18) & 0xFF
            dwidth = _SMEM_DATA_WIDTH.get(op)
            if dwidth is None:
                continue
            bwidth = 4 if (0x08 <= op <= 0x0F or 0x18 <= op <= 0x1F) else 2
            sdata = (w0 >> 6) & 0x7F
            sbase = (w0 & 0x3F) * 2
            d_hi = sdata <= 105 and 106 < sdata + dwidth <= 108
            b_hi = sbase <= 105 and 106 < sbase + bwidth <= 108
            if not (d_hi or b_hi):
                continue
            used = []
            if not d_hi:
                used.append((sdata, dwidth))
            if not b_hi:
                used.append((sbase, bwidth))
            if d_hi:
                ph = _pick_placeholder(dwidth, *used)
                used.append((ph, dwidth))
                pw0 = (pw0 & ~(0x7F << 6)) | ((ph & 0x7F) << 6)
                reps.append((f"s[{ph}:{ph + dwidth - 1}]", _vcc_tuple_str(sdata, dwidth)))
            if b_hi:
                ph = _pick_placeholder(bwidth, *used)
                pw0 = (pw0 & ~0x3F) | ((ph // 2) & 0x3F)
                reps.append((f"s[{ph}:{ph + bwidth - 1}]", _vcc_tuple_str(sbase, bwidth)))
        elif hi6 == 0x3C:  # MIMG (64-bit + optional NSA dwords)
            opc = ((w0 >> 18) & 0x7F) | ((w0 & 1) << 7)
            entry = rdna2_decode.T.MIMG.get(opc)
            if entry is None:
                continue
            name, spec = entry
            has_samp = any(fn == "SSAMP" for fn, _, _ in spec)
            rwidth = 4 if ((w0 & (1 << 15)) or name.startswith("image_bvh")) else 8
            srsrc = ((w1 >> 16) & 0x1F) * 4
            ssamp = ((w1 >> 21) & 0x1F) * 4
            r_hi = srsrc <= 105 and 106 < srsrc + rwidth <= 108
            s_hi = has_samp and ssamp <= 105 and 106 < ssamp + 4 <= 108
            if not (r_hi or s_hi):
                continue
            used = []
            if not r_hi:
                used.append((srsrc, rwidth))
            if has_samp and not s_hi:
                used.append((ssamp, 4))
            if r_hi:
                ph = _pick_placeholder(rwidth, *used)
                used.append((ph, rwidth))
                pw1 = (pw1 & ~(0x1F << 16)) | (((ph // 4) & 0x1F) << 16)
                reps.append((f"s[{ph}:{ph + rwidth - 1}]", _vcc_tuple_str(srsrc, rwidth)))
            if s_hi:
                ph = _pick_placeholder(4, *used)
                pw1 = (pw1 & ~(0x1F << 21)) | (((ph // 4) & 0x1F) << 21)
                reps.append((f"s[{ph}:{ph + 3}]", _vcc_tuple_str(ssamp, 4)))
        elif hi6 in (0x38, 0x3A):  # MUBUF / MTBUF (64-bit)
            srsrc = ((w1 >> 16) & 0x1F) * 4
            if not (srsrc <= 105 and 106 < srsrc + 4 <= 108):
                continue
            ph = 0
            pw1 = (pw1 & ~(0x1F << 16)) | (((ph // 4) & 0x1F) << 16)
            reps.append((f"s[{ph}:{ph + 3}]", _vcc_tuple_str(srsrc, 4)))
        else:
            continue
        patches[at] = (w0, w1, pw0, pw1, reps)
    return patches


def _run_mc(code: bytes, mcpu: str):
    text = " ".join(f"0x{b:02x}" for b in code)
    try:
        p = subprocess.run(MC + ["--disassemble", f"-triple={TRIPLE}", f"-mcpu={mcpu}", "-show-encoding"],
                           input=text, capture_output=True, text=True, errors="replace",
                           creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    except OSError as e:
        return "", 0, str(e)
    bad = p.stderr.count("invalid instruction encoding")
    if p.returncode != 0:
        return "", bad, p.stderr.strip() or f"llvm-mc exited with code {p.returncode}"
    return p.stdout, bad, p.stderr


def _rewrite_patched_asm(asm_text: str, patched: bytes, invalid_offs: set, patches: dict,
                         mcpu: str = "gfx1013", wave32: bool = True):
    """Map each disassembled line back to its byte offset in `patched`, restoring VCCTuple operands and original
    encoding bytes at patched offsets. Also repairs llvm-mc's dropped 4-byte literal on VOP3P instructions
    (e.g. `v_fma_mix_f32` with src=0xff, where llvm-mc consumes 12 bytes but prints 8 and replaces the literal
    with 0), which otherwise shifts downstream PCs and breaks branch targets."""
    pos, prev_pos, prev_idx, hit, lines = 0, -1, -1, set(), []
    for line in asm_text.split("\n"):
        enc = ENCODING.search(line)
        if not enc:
            lines.append(line)
            continue
        while pos in invalid_offs:
            pos += 4
        raw_enc = [b.strip() for b in enc.group(1).split(",")]
        enc_bytes = bytes(int(x, 16) for x in raw_enc)
        nbytes = len(enc_bytes)
        if patched[pos:pos + nbytes] != enc_bytes:
            for skip in (4, 8):
                if pos + skip + nbytes <= len(patched) and patched[pos + skip:pos + skip + nbytes] == enc_bytes:
                    if skip == 4 and prev_idx >= 0 and pos - prev_pos == 8 and (patched[prev_pos + 3] >> 2) == 0x33:
                        fixed, fbad = rdna2_decode.disasm_bytes(
                            patched[prev_pos:prev_pos + 12], mcpu=mcpu, wave32=wave32)
                        fixed_lines = [ln for ln in fixed.splitlines() if ENCODING.search(ln)]
                        if fbad == 0 and len(fixed_lines) == 1:
                            lines[prev_idx] = fixed_lines[0]
                    pos += skip
                    break
        if pos in patches and nbytes >= 8:
            orig_w0, orig_w1, _pw0, _pw1, reps = patches[pos]
            body = line[:enc.start()]
            ok = True
            for ph_text, real_text in reps:
                if ph_text not in body:
                    ok = False
                    break
                body = body.replace(ph_text, real_text, 1)
            if ok:
                hit.add(pos)
                orig_bytes = struct.pack("<2I", orig_w0, orig_w1)
                for idx, bval in enumerate(orig_bytes):
                    raw_enc[idx] = f"0x{bval:02x}"
                line = f"{body}; encoding: [{','.join(raw_enc)}]"
        prev_pos = pos
        prev_idx = len(lines)
        lines.append(line)
        pos += nbytes
    return "\n".join(lines), hit


def _with_sgpr_vcc_tuples(code: bytes, mcpu: str, asm: str, bad: int, err: str, wave32: bool = True):
    """Retry llvm-mc with aligned low-SGPR placeholders for rejected SMEM/MIMG/MUBUF encodings whose SGPR operand
    crosses s105 into vcc_lo/vcc_hi, then restore the VCCTuple operands and original encoding bytes."""
    patches = {}
    tried = set()
    while bad:
        invalid = {(int(col) - 1) // 5 for col in INVALID_AT.findall(err)}
        found = _sgpr_vcc_patches(code, invalid)
        new_keys = {k: v for k, v in found.items() if k not in tried}
        if not new_keys:
            break
        tried.update(new_keys)
        patches.update(new_keys)
        patched = bytearray(code)
        for at, (_w0, _w1, pw0, pw1, _reps) in patches.items():
            struct.pack_into("<2I", patched, at, pw0, pw1)
        patched_bytes = bytes(patched)
        asm2, bad2, err2 = _run_mc(patched_bytes, mcpu)
        if not asm2.strip():
            break
        invalid2 = {(int(col) - 1) // 5 for col in INVALID_AT.findall(err2)}
        rewritten, hit = _rewrite_patched_asm(asm2, patched_bytes, invalid2, patches, mcpu=mcpu, wave32=wave32)
        if hit != set(patches):
            # Drop any candidate that did not land on an instruction boundary and retry once without it.
            patches = {k: v for k, v in patches.items() if k in hit}
            if not patches:
                break
            patched = bytearray(code)
            for at, (_w0, _w1, pw0, pw1, _reps) in patches.items():
                struct.pack_into("<2I", patched, at, pw0, pw1)
            patched_bytes = bytes(patched)
            asm2, bad2, err2 = _run_mc(patched_bytes, mcpu)
            if not asm2.strip():
                break
            invalid2 = {(int(col) - 1) // 5 for col in INVALID_AT.findall(err2)}
            rewritten, _ = _rewrite_patched_asm(asm2, patched_bytes, invalid2, patches, mcpu=mcpu, wave32=wave32)
        asm, bad, err = rewritten, bad2, err2
    return asm, bad, err


def disasm(code: bytes, mcpu: str, backend: str = "llvm-mc", wave32: bool = True):
    if backend == "python":
        asm, bad = rdna2_decode.disasm_bytes(code, mcpu=mcpu, wave32=wave32)
        return asm, bad, ""
    asm, bad, err = _run_mc(code, mcpu)
    if asm.strip() and bad:
        asm, bad, err = _with_sgpr_vcc_tuples(code, mcpu, asm, bad, err, wave32=wave32)
    return asm, bad, err


MAX_DATA_DWORDS = 1 << 16


def data_block_start(b: bytes, start: int, end: int) -> int:
    """Where a constant-data block begins at the end of the code range [start, end), or `end` if there is none.

    Some blobs carry, after the last real s_endpgm, a block laid out as: s_endpgm, a dword count N, N dwords
    of data, then one more s_endpgm (the one the range ends on) and s_code_end padding. The data is not code:
    llvm-mc reports its words as invalid (and reads a 0xBF810000 inside it as a second program end). The
    block is recognised only when the count N matches the distance to the closing s_endpgm and the dword
    before the count is itself an s_endpgm, so ordinary code is not cut."""
    last = end - 4
    for n in range(min(MAX_DATA_DWORDS, (last - start) // 4 - 1) + 1):
        count = last - 4 * (n + 1)
        if count - 4 < start:
            break
        if struct.unpack_from("<I", b, count)[0] == n and b[count - 4:count] == ENDPGM:
            return count
    return end


def code_range(b: bytes):
    """The code section offset is the u32 at 0xC0; instructions start 0x28 bytes into it and run to the last
    4-byte-aligned s_endpgm in the blob (some blobs hold several programs), minus a trailing constant-data
    block (see data_block_start). Found by correlating against clean decodes of 154 random shaders."""
    # A blob too short to hold the offset goes to the scan like any other header miss; reading it anyway raised
    # struct.error, which stopped the whole run (and make_game.py at the same blob on every rerun).
    off = struct.unpack_from("<I", b, 0xC0)[0] if len(b) >= 0xC4 else 0
    start = off + 0x28
    if 0 < off and start < len(b):
        last = b.rfind(ENDPGM, start)
        while last >= 0 and (last - start) % 4:
            last = b.rfind(ENDPGM, start, last)
        if last >= 0:
            return start, data_block_start(b, start, last + 4), "header"
    end = b.find(ENDPGM)
    # Fallback instructions are dword-aligned in the blob. A matching byte sequence can straddle metadata words.
    while end >= 0 and end % 4:
        end = b.find(ENDPGM, end + 1)
    return None, end + 4 if end >= 0 else None, "scan"


def one(path, out, mcpu, backend="llvm-mc"):
    with open(path, "rb") as f:
        b = f.read()
    wave32 = fssm_header.wave_size(b) != 64
    start, end, how = code_range(b)
    if end is None:
        return "no s_endpgm"
    if start is None:
        # Walk back in 4-byte steps from the end and keep the longest clean decode.
        start, best = end - 4, None
        for s in range(end - 4, max(0x100, end - 0x4000) - 1, -4):
            _, bad, _ = disasm(b[s:end], mcpu, backend=backend, wave32=wave32)
            if bad == 0:
                best = s
        if best is None:
            return "scan failed"
        start = best
    asm, bad, err = disasm(b[start:end], mcpu, backend=backend, wave32=wave32)
    if not asm.strip():  # a broken toolchain, not a shader: never report this as "clean"
        reason = (err.strip().splitlines() or ["no output"])[0]
        FAILED.append((path, reason))
        return "failed"
    name = os.path.splitext(os.path.basename(path))[0]
    # Only a published listing can be reused after an interruption.
    listing = tempfile.NamedTemporaryFile(mode="w", dir=out, prefix=".disasm-", suffix=".tmp",
                                          encoding="utf-8", newline="\n", delete=False)
    try:
        with listing:
            listing.write(f"; {name}  code 0x{start:x}..0x{end:x} ({how}), {mcpu}, {bad} invalid words\n")
            listing.write(asm)
            if err.strip():
                listing.write("\n; --- llvm-mc messages ---\n; " + err.strip().replace("\n", "\n; ") + "\n")
        os.replace(listing.name, output_path(path, out))
    finally:
        if os.path.exists(listing.name):
            os.unlink(listing.name)
    return "clean" if bad == 0 else "invalid words"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--unpack-index", help="disassemble only the FSSMs listed in this unpack index")
    ap.add_argument("--out")
    ap.add_argument("--skip-existing", action="store_true", help="reuse complete listings from a matching run")
    ap.add_argument("--mcpu", default="gfx1013")
    ap.add_argument("--backend", choices=("llvm-mc", "python"), default="llvm-mc",
                    help="disassembler backend: llvm-mc (default) or pure-Python RDNA2 decoder (#1901)")
    ap.add_argument("--limit", type=positive_int)
    ap.add_argument("--jobs", type=positive_int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    if os.path.isfile(a.src):
        if a.unpack_index:
            ap.error("--unpack-index requires an FSSM directory")
        files = [a.src]
    elif os.path.isdir(a.src):
        try:
            if a.unpack_index:
                names = shader_stems(a.unpack_index, "name")
                files = sorted(os.path.join(a.src, name + ".fssm") for name in names)
                for path in files:
                    if not os.path.isfile(path):
                        raise ValueError(f"{a.unpack_index}: declared FSSM file is missing: {path}")
            else:
                with os.scandir(a.src) as entries:
                    files = sorted(entry.path for entry in entries
                                   if entry.name.endswith(".fssm") and entry.is_file())
        except (OSError, ValueError) as e:
            fail(f"cannot list shaders in {a.src}: {e}")
    elif not os.path.exists(a.src):
        print(f"ERROR: {a.src}: does not exist", file=sys.stderr)
        return 2
    else:
        fail(f"{a.src} is not a file or folder")
    if not files:
        fail(f"no .fssm files in {a.src}")
    files = files[: a.limit] if a.limit else files
    out = a.out or os.path.join(os.path.dirname(files[0]), "..", "disasm")
    validate_output(out, files)
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        fail(f"cannot create the output folder {out}: {e.strerror or e}")
    from collections import Counter
    res = Counter()
    total = len(files)
    every_stem = {os.path.splitext(os.path.basename(p))[0] for p in files}   # before --skip-existing narrows files
    if a.skip_existing:
        pending = []
        for path in files:
            listing = output_path(path, out)
            if not os.path.islink(listing) and os.path.isfile(listing) and os.path.getsize(listing) > 0:
                res["reused"] += 1
            else:
                pending.append(path)
        files = pending
    if files:
        if a.backend == "llvm-mc":
            probe(a.mcpu)
        with ThreadPoolExecutor(a.jobs) as ex:
            res.update(ex.map(lambda p: one(p, out, a.mcpu, backend=a.backend), files))
    print(f"{total} shaders -> {os.path.abspath(out)}: {dict(res)}")
    if FAILED:
        path, reason = FAILED[0]
        fail(f"{len(FAILED)} shader(s) produced no disassembly, first {path}: {reason}")
    if not (res["clean"] or res["invalid words"] or res["reused"]):
        fail(f"no disassembly produced for {a.src}")
    if os.path.isdir(a.src) and not a.limit:
        # A shader dropped by a re-carve must not keep its old listing: the translator reads every .s in this folder,
        # so a stale one would still be translated and indexed even though the corpus no longer has it. Prune only
        # after a successful decode, so a missing llvm-mc or a failed shader never deletes a usable listing. --limit
        # is a partial run on purpose and leaves the rest of the folder alone.
        # Every input's stem, not just this run's `files`: --skip-existing leaves the reused listings out of `files`,
        # and pruning against that list deleted every listing a resumed run had just reused.
        stems = every_stem
        stale = []
        try:
            with os.scandir(out) as entries:
                stale = sorted(e.path for e in entries
                               if e.name.endswith(".s") and e.is_file() and e.name[:-2] not in stems)
            for path in stale:
                os.remove(path)
        except OSError as e:
            fail(f"cannot prune stale .s files in {out}: {e.strerror or e}")
        if stale:
            print(f"removed {len(stale)} stale .s file(s) that no .fssm in {a.src} names")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
