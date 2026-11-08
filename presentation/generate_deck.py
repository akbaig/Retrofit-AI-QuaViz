#!/usr/bin/env python3
"""Generate a clean image-forward presentation deck for Retrofit-QuaViz video."""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

COLOR_BG = RGBColor(250, 250, 252)
COLOR_TITLE = RGBColor(15, 23, 42)
COLOR_MUTED = RGBColor(100, 116, 139)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG


def add_title(slide, title, y=Inches(0.28)):
    box = slide.shapes.add_textbox(Inches(0.6), y, Inches(12.1), Inches(0.55))
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Arial"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = COLOR_TITLE
    p.alignment = PP_ALIGN.CENTER


def add_caption(slide, text, y=Inches(7.05)):
    box = slide.shapes.add_textbox(Inches(0.6), y, Inches(12.1), Inches(0.35))
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_MUTED
    p.alignment = PP_ALIGN.CENTER


def fit_image(path, max_w, max_h):
    with Image.open(path) as im:
        iw, ih = im.size
    aspect = iw / ih
    max_w_in = max_w.inches if hasattr(max_w, "inches") else max_w
    max_h_in = max_h.inches if hasattr(max_h, "inches") else max_h
    if max_w_in / max_h_in > aspect:
        h = max_h_in
        w = h * aspect
    else:
        w = max_w_in
        h = w / aspect
    return Inches(w), Inches(h)


def place_centered(slide, path, max_w, max_h, top):
    w, h = fit_image(path, max_w, max_h)
    left = (SLIDE_W - w) / 2
    slide.shapes.add_picture(path, left, top, w, h)
    return w, h


def place_at(slide, path, left, top, max_w, max_h):
    w, h = fit_image(path, max_w, max_h)
    slide.shapes.add_picture(path, left, top, w, h)
    return w, h


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def build(workspace, out_path):
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    img = {
        "overall": os.path.join(workspace, "presentation_images", "01_overall.jpg"),
        "workflow": os.path.join(workspace, "presentation_images", "workflow.jpeg"),
        "composite": os.path.join(workspace, "submissions", "01_training", "figures", "01_headline_composites.png"),
        "val": os.path.join(workspace, "submissions", "01_training", "figures", "02_per_type_validation.png"),
        "holdout": os.path.join(workspace, "submissions", "01_training", "figures", "02_per_type_holdout.png"),
        "system": os.path.join(workspace, "presentation_images", "overall_system.jpeg"),
        "edge": os.path.join(workspace, "submissions", "04_edge", "edge_design.png"),
        "demo": os.path.join(workspace, "presentation_images", "07_conclusion.jpg"),
    }

    # ── Slide 1: Problem / Overall ──────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    place_centered(s, img["overall"], Inches(12.5), Inches(6.6), Inches(0.45))
    add_notes(s,
        "Industrial machines already generate valuable data. What they often lack is a way "
        "to turn those signals into early warnings and useful maintenance decisions.\n\n"
        "Our challenge was to do that for an existing cooling-water pump: detect developing "
        "faults, explain the evidence, and recommend what an operator should do next—"
        "without controlling the equipment.\n\nWe built Retrofit-QuaViz.")

    # ── Slide 2: ML Approach ────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    place_centered(s, img["workflow"], Inches(12.5), Inches(6.6), Inches(0.45))
    add_notes(s,
        "The system starts with 14 tree-based regressors that estimate how signals should "
        "behave under healthy conditions. We compare those predictions with actual "
        "measurements, then pass the deviations to an ensemble of boosted-tree classifiers "
        "that score the possible fault types.")

    # ── Slide 3: Headline composite ─────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_title(s, "Official Validation Performance")
    place_centered(s, img["composite"], Inches(11.5), Inches(5.8), Inches(1.0))
    add_notes(s,
        "In our evaluation, our approach correctly detected three additional fault types "
        "that the best provided baseline failed to detect.")

    # ── Slide 4: Per-type validation + holdout side by side ─────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_title(s, "Per-Fault Detection Accuracy")

    gap = Inches(0.35)
    max_w = Inches(6.0)
    max_h = Inches(5.8)
    w1, h1 = fit_image(img["val"], max_w, max_h)
    w2, h2 = fit_image(img["holdout"], max_w, max_h)
    row_h = max(h1, h2)
    total_w = w1 + gap + w2
    left1 = (SLIDE_W - total_w) / 2
    left2 = left1 + w1 + gap
    top = Inches(0.95)
    s.shapes.add_picture(img["val"], left1, top + (row_h - h1) / 2, w1, h1)
    s.shapes.add_picture(img["holdout"], left2, top + (row_h - h2) / 2, w2, h2)
    add_notes(s,
        "QuaViz achieved 100% first-alert detection on Bearing Wear, Misalignment, "
        "and Sensor Drift — faults the best baseline missed entirely. "
        "Transfer holds across 78 post-freeze holdout events.")

    # ── Slide 4: Overall System ─────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    place_centered(s, img["system"], Inches(12.5), Inches(6.6), Inches(0.45))
    add_notes(s,
        "We deploy the frozen model on a backend that processes incoming telemetry. "
        "Fault-specific thresholds and persistence rules turn continuous scores into "
        "actionable alerts while limiting nuisance notifications.\n\n"
        "Each alert includes the predicted fault, its confidence score, and the "
        "measurements supporting it. An LLM then turns that structured evidence into a "
        "concise explanation, drawing recommendations from a predefined action catalog. "
        "It cannot change the diagnosis or alert status.\n\n"
        "Operators can inspect the evidence, review the recommendation, and use the "
        "sandbox to explore the potential effects of an intervention. The decision to "
        "act remains with a human; the system sends no commands to plant equipment.")

    # ── Slide 5: Edge Design ────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    place_centered(s, img["edge"], Inches(12.5), Inches(6.6), Inches(0.45))
    add_notes(s,
        "The architecture also separates the machinery-specific models from the "
        "explanation and interface layers. That gives us a reusable foundation for other "
        "equipment, with appropriate retraining, configuration, and validation. Cloud or "
        "local explanations allow the system to accommodate different connectivity "
        "requirements.")

    # ── Slide 6: Demo / In Action ───────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_title(s, "Retrofit-QuaViz in Action")
    place_centered(s, img["demo"], Inches(11.5), Inches(5.6), Inches(1.05))
    add_caption(s, "Demo video follows")
    add_notes(s, "Now, let’s see Retrofit-QuaViz in action.")

    prs.save(out_path)
    print(f"Saved: {out_path} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.abspath(os.path.join(base, ".."))
    out = os.path.join(base, "Retrofit_QuaViz_Presentation.pptx")
    build(workspace, out)
