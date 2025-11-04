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
	outputs.append(build_and_draw(yaml_l, "RT-DETR-L", out_dir))
	outputs.append(build_and_draw(yaml_mnv4, "RT-DETR-MNV4-Hybrid-M", out_dir))
	outputs.append(build_and_draw(yaml_mnv4_sea, "RT-DETR-MNV4-Hybrid-M-SEA", out_dir))

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

