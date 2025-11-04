import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
	import yaml  # type: ignore
except Exception:
	yaml = None  # deferred import error handled in main

try:
	import svgwrite  # type: ignore
except Exception:
	svgwrite = None  # deferred import error handled in main


# -----------------------------
# Utilities to parse RT-DETR YAML
# -----------------------------

BackboneItem = Tuple[int, int, str, List[Any]]


def load_yaml(path: Path) -> Dict[str, Any]:
	if yaml is None:
		raise RuntimeError("Missing dependency: PyYAML. Please install 'pyyaml'.")
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def to_module_tuple(item: List[Any]) -> BackboneItem:
	"""Convert a YAML backbone/head line to (from, repeats, module, args)."""
	assert isinstance(item, list) and len(item) >= 4, f"Invalid item: {item}"
	from_idx = item[0] if isinstance(item[0], int) else -1
	repeats = int(item[1])
	module = item[2] if isinstance(item[2], str) else str(item[2])
	args = item[3] if isinstance(item[3], list) else []
	return from_idx, repeats, module, args


def infer_stride(module: str, args: List[Any]) -> int:
	"""Infer spatial stride contribution for backbone blocks (downsample only)."""
	m = module
	if m in ("Conv", "DWConv"):
		# [out, k, s, ...]
		if len(args) >= 3 and isinstance(args[2], int):
			return int(args[2])
		return 1
	if m in ("EdgeResidual", "UniversalInvertedResidual", "HGStem", "HGBlock"):
		# common pattern: second arg is stride for IR-like blocks; HGBlock uses DWConv elsewhere
		if len(args) >= 2 and isinstance(args[1], int):
			return int(args[1])
		return 1
	# other modules: assume stride 1 for backbone
	return 1


def infer_channels(module: str, args: List[Any], cin: int) -> int:
	"""Infer output channels for a block; fallback to input channels if unknown."""
	m = module
	if m in ("Conv", "DWConv", "EdgeResidual", "UniversalInvertedResidual", "C2f", "HGBlock", "HGStem"):
		if len(args) >= 1 and isinstance(args[0], int):
			return int(args[0])
	# AIFI, nn.Upsample, Concat, RepC3 etc: keep channels or deduce externally
	return cin


def summarize_backbone(backbone: List[List[Any]], input_size: int = 640) -> List[Dict[str, Any]]:
	"""Aggregate backbone into stages by spatial size; mark SEA modules and collect brief labels."""
	H = W = input_size
	C = 3
	stages: List[Dict[str, Any]] = []

	def stage_name_from_stride(total_down: int) -> str:
		# P1/2=320, P2/4=160, P3/8=80, P4/16=40, P5/32=20
		mapping = {1: "P1/2", 2: "P2/4", 3: "P3/8", 4: "P4/16", 5: "P5/32"}
		return mapping.get(total_down, f"P?/{2**total_down}")

	total_down = 0
	current_group: List[str] = []
	current_has_sea = False
	current_c = C
	current_hw: Tuple[int, int] = (H, W)

	def flush_group():
		nonlocal current_group, current_has_sea, current_c, current_hw, total_down
		if not current_group:
			return
		stages.append(
			{
				"name": stage_name_from_stride(total_down),
				"hw": f"{current_hw[0]}x{current_hw[1]}",
				"c": current_c,
				"modules": list(current_group),
				"sea": current_has_sea,
			}
		)
		current_group = []
		current_has_sea = False

	# Iterate backbone
	for item in backbone:
		from_idx, repeats, module, args = to_module_tuple(item)
		# Expand repeats for summary granularity
		for _ in range(repeats):
			stride = infer_stride(module, args)
			out_c = infer_channels(module, args, C)
			label = module
			if module.startswith("nn."):
				label = module.replace("nn.", "")
			if "SEA" in module or "Sea_" in module or "Sea" in module:
				current_has_sea = True
				label = f"{label} [SEA]"
			# If this block changes resolution (downsample), we close previous group and start new stage
			if stride and stride > 1:
				# flush previous group (belongs to previous stage)
				flush_group()
				# apply downsample
				H //= stride
				W //= stride
				total_down += 1
				current_hw = (H, W)
			# Update channels and collect label
			C = out_c
			current_c = C
			current_group.append(label)

	# flush tail
	flush_group()
	return stages


# -----------------------------
# SVG rendering
# -----------------------------


def draw_model_svg(
	out_path: Path,
	title: str,
	stages: List[Dict[str, Any]],
	head_summary: List[str],
	canvas_size: Tuple[int, int] = (1200, 800),
) -> None:
	if svgwrite is None:
		raise RuntimeError("Missing dependency: svgwrite. Please install 'svgwrite'.")

	dwg = svgwrite.Drawing(str(out_path), size=(canvas_size[0], canvas_size[1]))

	# Styles
	BG = "#ffffff"
	COLOR_STAGE = "#F4F5F7"
	COLOR_STAGE_SEA = "#E6F0FF"
	COLOR_HEAD = "#FFF7E6"
	COLOR_TITLE = "#172B4D"
	COLOR_TEXT = "#091E42"
	COLOR_ACCENT = "#4C9AFF"

	dwg.add(dwg.rect(insert=(0, 0), size=canvas_size, fill=BG))

	# Title
	dwg.add(
		dwg.text(
			title,
			insert=(20, 40),
			fill=COLOR_TITLE,
			font_size=24,
			font_weight="bold",
		)
	)

	# Layout params
	x = 20
	y = 70
	box_width = 360
	box_height_unit = 20  # each module line adds this height
	v_gap = 14
	h_gap = 20

	# Draw backbone stages
	for st in stages:
		modules = st.get("modules", [])
		mod_lines = min(8, len(modules))  # limit lines visible
		box_h = 80 + mod_lines * box_height_unit
		stage_color = COLOR_STAGE_SEA if st.get("sea") else COLOR_STAGE
		# Box
		dwg.add(
			dwg.rect(
				insert=(x, y),
				size=(box_width, box_h),
				rx=8,
				ry=8,
				fill=stage_color,
				stroke="#C1C7D0",
			)
		)
		# Header text
		header = f"{st['name']}  {st['hw']}  C={st['c']}"
		dwg.add(dwg.text(header, insert=(x + 12, y + 24), fill=COLOR_TEXT, font_size=14, font_weight="bold"))
		# Module list
		ty = y + 48
		for i, m in enumerate(modules[:mod_lines]):
			fill = COLOR_ACCENT if "[SEA]" in m else COLOR_TEXT
			dwg.add(dwg.text(f"• {m}", insert=(x + 12, ty), fill=fill, font_size=12))
			ty += box_height_unit
		# More indicator
		if len(modules) > mod_lines:
			dwg.add(dwg.text(f"… (+{len(modules)-mod_lines} blocks)", insert=(x + 12, ty), fill=COLOR_TEXT, font_size=12, opacity=0.7))
		y += box_h + v_gap

	# Head summary panel
	y_head = 70
	x_head = x + box_width + h_gap
	head_h = 140 + len(head_summary) * box_height_unit
	dwg.add(
		dwg.rect(insert=(x_head, y_head), size=(box_width, head_h), rx=8, ry=8, fill=COLOR_HEAD, stroke="#E6C200")
	)
	dwg.add(dwg.text("Head (AIFI + FPN/PAN + Decoder)", insert=(x_head + 12, y_head + 24), fill=COLOR_TEXT, font_size=14, font_weight="bold"))
	ty = y_head + 48
	for line in head_summary:
		dwg.add(dwg.text(f"• {line}", insert=(x_head + 12, ty), fill=COLOR_TEXT, font_size=12))
		ty += box_height_unit

	dwg.save()


def make_head_summary(model_name: str) -> List[str]:
	# Generic RT-DETR head summary for P3/P4/P5
	lines = [
		"P5: 20x20xC5 → Conv1x1(256) → AIFI → Conv1x1(256) = Y5(20x20x256)",
		"P4: Upsample(Y5 20→40) + Stage3(40x40)投影(→256) → RepC3 → Y4(40x40x256)",
		"P3: Upsample(Y4 40→80) + Stage2(80x80)投影(→256) → RepC3 → X3(80x80x256)",
		"PAN: X3↓→40 + Y4 → F4；F4↓→20 + Y5 → F5",
		"Decoder: RTDETRDecoder([X3, F4, F5])",
	]
	return lines


def draw_compare_svg(
	out_path: Path,
	titles: List[str],
	models_stages: List[List[Dict[str, Any]]],
	canvas_size: Tuple[int, int] = (1500, 900),
) -> None:
	"""Draw a side-by-side comparison focusing on backbone replacement and SEA insertion."""
	if svgwrite is None:
		raise RuntimeError("Missing dependency: svgwrite. Please install 'svgwrite'.")

	assert len(titles) == len(models_stages) == 3, "Expect exactly three models for comparison"

	dwg = svgwrite.Drawing(str(out_path), size=(canvas_size[0], canvas_size[1]))

	BG = "#ffffff"
	COLOR_STAGE = "#F4F5F7"
	COLOR_STAGE_SEA = "#E6F0FF"
	COLOR_TEXT = "#091E42"
	COLOR_TITLE = "#172B4D"
	COLOR_NOTE = "#6554C0"
	COLOR_SEP = "#C1C7D0"
	COLOR_ACCENT = "#4C9AFF"

	dwg.add(dwg.rect(insert=(0, 0), size=canvas_size, fill=BG))
	dwg.add(
		dwg.text(
			"RT-DETR 主干对比：L vs MNV4 vs MNV4+SEA（蓝色=含SEA）",
			insert=(20, 40),
			fill=COLOR_TITLE,
			font_size=24,
			font_weight="bold",
		)
	)

	# Column layout
	cols = 3
	margin_x = 24
	margin_top = 70
	col_w = (canvas_size[0] - margin_x * 2) / cols
	box_width = col_w - 20
	box_height_unit = 18

	# For each column
	for ci in range(cols):
		x = margin_x + ci * col_w + 10
		y = margin_top + 10
		# Title per column
		dwg.add(dwg.text(titles[ci], insert=(x, y), fill=COLOR_TITLE, font_size=18, font_weight="bold"))
		y += 24
		stages = models_stages[ci]
		for st in stages:
			modules = st.get("modules", [])
			mod_lines = min(7, len(modules))
			box_h = 70 + mod_lines * box_height_unit
			stage_color = COLOR_STAGE_SEA if st.get("sea") else COLOR_STAGE
			dwg.add(
				dwg.rect(
					insert=(x, y),
					size=(box_width, box_h),
					rx=8,
					ry=8,
					fill=stage_color,
					stroke=COLOR_SEP,
				)
			)
			header = f"{st['name']}  {st['hw']}  C={st['c']}"
			dwg.add(dwg.text(header, insert=(x + 10, y + 22), fill=COLOR_TEXT, font_size=13, font_weight="bold"))
			ty = y + 44
			for m in modules[:mod_lines]:
				fill = COLOR_ACCENT if "[SEA]" in m else COLOR_TEXT
				dwg.add(dwg.text(f"• {m}", insert=(x + 10, ty), fill=fill, font_size=12))
				ty += box_height_unit
			if len(modules) > mod_lines:
				dwg.add(dwg.text(f"… (+{len(modules)-mod_lines})", insert=(x + 10, ty), fill=COLOR_TEXT, font_size=12, opacity=0.7))
			y += box_h + 12

		# Column separator
		if ci < cols - 1:
			x_sep = margin_x + (ci + 1) * col_w
			dwg.add(dwg.line(start=(x_sep, margin_top), end=(x_sep, canvas_size[1] - 40), stroke=COLOR_SEP, stroke_width=1))

	# Shared head note (unchanged across columns)
	note = "Head 保持一致：AIFI + FPN/PAN + RTDETRDecoder（对比重点在主干替换与注意力插入）"
	dwg.add(dwg.text(note, insert=(20, canvas_size[1] - 24), fill=COLOR_NOTE, font_size=14))

	dwg.save()


def build_and_draw(model_yaml: Path, title: str, out_dir: Path) -> Path:
	data = load_yaml(model_yaml)
	backbone = data.get("backbone", [])
	stages = summarize_backbone(backbone)
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{title.replace(' ', '_').lower()}.svg"
	head_summary = make_head_summary(title)
	draw_model_svg(out_path, title, stages, head_summary)
	return out_path


def main():
	repo_root = Path(__file__).resolve().parents[2]
	models_dir = repo_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "rt-detr"
	out_dir = repo_root / "docs" / "figures" / "rtdetr_arch"

	# Model YAMLs
	yaml_l = models_dir / "rtdetr-l.yaml"
	yaml_mnv4 = models_dir / "rtdetr-mnv4-hybrid-m.yaml"
	yaml_mnv4_sea = models_dir / "rtdetr-mnv4-hybrid-m-sea.yaml"

	missing: List[str] = [str(p) for p in [yaml_l, yaml_mnv4, yaml_mnv4_sea] if not p.exists()]
	if missing:
		raise FileNotFoundError(f"Missing YAML files: {missing}")

	outputs: List[Path] = []
	# Individual diagrams (SVG) — 如果缺少 svgwrite，跳过 SVG 生成
	try:
		p_l = build_and_draw(yaml_l, "RT-DETR-L", out_dir)
		p_m = build_and_draw(yaml_mnv4, "RT-DETR-MNV4-Hybrid-M", out_dir)
		p_ms = build_and_draw(yaml_mnv4_sea, "RT-DETR-MNV4-Hybrid-M-SEA", out_dir)
		outputs.extend([p_l, p_m, p_ms])
	except RuntimeError as e:
		if "svgwrite" in str(e):
			print("[Info] svgwrite 未安装，跳过 SVG 生成，仅生成 ASCII/DOT。")
		else:
			raise

	# Comparison diagram (backbone only focus)
	data_l = load_yaml(yaml_l)
	data_m = load_yaml(yaml_mnv4)
	data_ms = load_yaml(yaml_mnv4_sea)
	stages_l = summarize_backbone(data_l.get("backbone", []))
	stages_m = summarize_backbone(data_m.get("backbone", []))
	stages_ms = summarize_backbone(data_ms.get("backbone", []))
	# 对比 SVG（若 svgwrite 缺失则跳过）
	try:
		compare_out = out_dir / "rtdetr_compare.svg"
		draw_compare_svg(compare_out, ["RT-DETR-L", "MNV4-Hybrid-M", "MNV4-Hybrid-M-SEA"], [stages_l, stages_m, stages_ms])
		outputs.append(compare_out)
	except RuntimeError as e:
		if "svgwrite" in str(e):
			print("[Info] svgwrite 未安装，跳过对比 SVG 生成。")
		else:
			raise

	# -----------------------------
	# ASCII generation
	# -----------------------------
	def ascii_block(title: str, stages: List[Dict[str, Any]]) -> str:
		lines: List[str] = []
		lines.append(f"# {title}")
		lines.append("")
		lines.append("Input: 640x640x3")
		for i, st in enumerate(stages):
			sea_mark = " [SEA]" if st.get("sea") else ""
			lines.append(f"└─ {st['name']}  {st['hw']}  C={st['c']}{sea_mark}")
			mods = st.get("modules", [])
			for m in mods:
				prefix = "   ├─" if m != mods[-1] else "   └─"
				lines.append(f"{prefix} {m}")
		# Generic head note
		lines.append("")
		lines.append("Head: AIFI + FPN/PAN + RTDETRDecoder (P3/8, P4/16, P5/32)")
		return "\n".join(lines)

	ascii_dir = out_dir
	(ascii_dir / "ascii").mkdir(parents=True, exist_ok=True)
	ascii_l = ascii_dir / "ascii" / "rtdetr_l_ascii.md"
	ascii_m = ascii_dir / "ascii" / "rtdetr_mnv4_hybrid_m_ascii.md"
	ascii_ms = ascii_dir / "ascii" / "rtdetr_mnv4_hybrid_m_sea_ascii.md"
	ascii_cmp = ascii_dir / "ascii" / "rtdetr_compare_ascii.md"

	with open(ascii_l, "w", encoding="utf-8") as f:
		f.write("```\n" + ascii_block("RT-DETR-L", stages_l) + "\n```\n")
	with open(ascii_m, "w", encoding="utf-8") as f:
		f.write("```\n" + ascii_block("RT-DETR-MNV4-Hybrid-M", stages_m) + "\n```\n")
	with open(ascii_ms, "w", encoding="utf-8") as f:
		f.write("```\n" + ascii_block("RT-DETR-MNV4-Hybrid-M-SEA", stages_ms) + "\n```\n")
	with open(ascii_cmp, "w", encoding="utf-8") as f:
		f.write("# RT-DETR 主干对比 (L / MNV4 / MNV4+SEA)\n\n")
		f.write("## L\n\n```\n" + ascii_block("RT-DETR-L", stages_l) + "\n```\n\n")
		f.write("## MNV4\n\n```\n" + ascii_block("RT-DETR-MNV4-Hybrid-M", stages_m) + "\n```\n\n")
		f.write("## MNV4+SEA\n\n```\n" + ascii_block("RT-DETR-MNV4-Hybrid-M-SEA", stages_ms) + "\n```\n")

	outputs.extend([ascii_l, ascii_m, ascii_ms, ascii_cmp])

	# -----------------------------
	# Graphviz DOT generation
	# -----------------------------
	def dot_for_model(title: str, stages: List[Dict[str, Any]]) -> str:
		def esc(s: str) -> str:
			return s.replace("\"", "\\\"")
		lines: List[str] = []
		lines.append("digraph G {")
		lines.append("  rankdir=LR;")
		lines.append("  node [shape=box, style=filled, fontname=Helvetica];")
		# clusters per stage
		for i, st in enumerate(stages):
			color = "#E6F0FF" if st.get("sea") else "#F4F5F7"
			lines.append(f"  subgraph cluster_{i} {{")
			lines.append(f"    label=\"{esc(st['name'])} {esc(st['hw'])} C={st['c']}\";")
			lines.append(f"    color=\"#C1C7D0\";")
			lines.append(f"    style=filled; fillcolor=\"{color}\";")
			# one node per stage with module list
			mods = st.get("modules", [])
			mod_text = "\\n".join([m for m in mods[:10]])
			if len(mods) > 10:
				mod_text += f"\\n… (+{len(mods)-10})"
			lines.append(f"    s{i} [label=\"{esc(mod_text)}\"];\n  }}")
		# stage edges
		for i in range(len(stages) - 1):
			lines.append(f"  s{i} -> s{i+1};")
		# title node
		lines.append(f"  labelloc=\"t\"; label=\"{esc(title)} (Backbone)\";")
		lines.append("}")
		return "\n".join(lines)

	dot_dir = out_dir / "dot"
	dot_dir.mkdir(parents=True, exist_ok=True)
	dot_l = dot_dir / "rtdetr_l.dot"
	dot_m = dot_dir / "rtdetr_mnv4_hybrid_m.dot"
	dot_ms = dot_dir / "rtdetr_mnv4_hybrid_m_sea.dot"
	with open(dot_l, "w", encoding="utf-8") as f:
		f.write(dot_for_model("RT-DETR-L", stages_l))
	with open(dot_m, "w", encoding="utf-8") as f:
		f.write(dot_for_model("RT-DETR-MNV4-Hybrid-M", stages_m))
	with open(dot_ms, "w", encoding="utf-8") as f:
		f.write(dot_for_model("RT-DETR-MNV4-Hybrid-M-SEA", stages_ms))
	outputs.extend([dot_l, dot_m, dot_ms])

	# Optional: combined note file listing paths
	note = out_dir / "_index.txt"
	with open(note, "w", encoding="utf-8") as f:
		f.write("Generated SVGs:\n")
		for p in outputs:
			f.write(str(p) + "\n")

	print("SVGs generated:")
	for p in outputs:
		print(" -", p)


if __name__ == "__main__":
	main()

