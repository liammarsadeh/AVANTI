# AVANTI — AI-Powered Visual Assistive Navigation & Task Independence

> A chest-mounted wearable AI system that helps visually impaired users navigate indoor spaces, avoid obstacles, read text, and interact with an AI assistant — hands-free, entirely through voice and haptic feedback.

Developed at the **Faculty of Information Technology, American University of Madaba** (Graduation Project, 2026).

**Authors:** Saif Al Sakit, Ammar Sadeh
**Supervisor:** Prof. Mohammad Abbadi

---

## Overview

Over 253 million people worldwide live with visual impairment, and 39 million are completely blind. Existing assistive tools tend to specialize — white canes and guide dogs support only short-range obstacle avoidance, while smart glasses offer OCR and basic AI assistance but little real-time navigation support. **AVANTI** was built to close this gap: a single wearable device that combines obstacle detection, indoor mapping, text reading, and conversational AI assistance in one low-cost package.

The device is worn on the chest, which improves stability, reduces the chance of falling, and gives the camera a clearer view of the environment compared to head- or wrist-mounted designs.

## Key Features

- 🧭 **Indoor navigation** — real-time route planning and obstacle-aware path recalculation via SLAM
- 🚧 **Obstacle detection** — custom-trained YOLO model tuned for indoor hazards (doors, stairs, glass doors, poles, elevators, escalators) in addition to standard COCO object classes
- 📏 **Depth sensing** — monocular distance estimation, adapted from the original stereo-depth design when the specified hardware wasn't available
- 📖 **Text reading (OCR)** — reads signs, labels, documents, and handwritten notes aloud
- 🎙️ **Voice-driven AI assistant** — natural spoken interaction for questions, translation, and information retrieval
- 📳 **Dual feedback channels** — synchronized audio narration and haptic vibration alerts, so users aren't dependent on hearing alone

## System Architecture

A Raspberry Pi 5 acts as the central controller, coordinating two parallel pipelines:

| Pipeline | Flow |
|---|---|
| **Walking (navigation)** | Depth camera → OpenCV preprocessing → YOLO object detection → SLAM mapping & localization → obstacle decision gate → haptic + voice alert |
| **User command (AI assistant)** | Microphone → Faster-Whisper (STT) → AI assistant (intent classification) → OCR / translation / search / navigation module → Piper (TTS) → audio output |

## Tech Stack

**Hardware**
- Raspberry Pi 5 (8GB RAM, quad-core ARM Cortex-A76)
- RGB camera for perception and monocular depth estimation
- Grove haptic vibration motor (ERM, I²C, DRV2605L driver)
- Bluetooth audio output (earbuds)

**Software / AI**
| Component | Library / Model |
|---|---|
| Object detection | Custom-trained **YOLOv26s** |
| Depth estimation | **Depth Anything V2** (monocular) |
| Mapping & localization | **ORB-SLAM3** (via ROS) |
| Image preprocessing | **OpenCV** |
| Text recognition | **EasyOCR** |
| Speech-to-text | **Faster-Whisper** |
| Language understanding | **OpenAI LLM** |
| Text-to-speech | **Piper** |
| Model training | **PyTorch / Ultralytics** |

> **Note on hardware substitution:** the original design specified an Intel RealSense D435i stereo depth camera. That component could not be sourced during implementation, so the perception pipeline was re-engineered around Depth Anything V2 for monocular depth estimation — the rest of the obstacle-avoidance logic was unaffected.

## Dataset

The obstacle-detection dataset merges multiple sources into one YOLO-format training set:

- **HomeObjects-3K** — base indoor object dataset
- **Roboflow** — supplementary indoor annotations
- **Open Images V7** — targeted class extraction via FiftyOne
- **Custom-labeled classes** — doors, stairs, glass doors, poles, elevators, escalators (not well represented in public datasets)

A custom label-remapping and class-balancing pipeline was built to handle multi-source merging safely — since YOLO reads only numeric class indices, merging sources with independently defined class lists silently mislabels data unless indices are explicitly remapped. Class balance was managed with image-count-based, rarest-class-first sampling rather than instance-count targeting.

## Results

### Object Detection Performance

| Metric | Value |
|---|---|
| mAP@50 | **74.6%** |
| mAP@50–95 | **60.0%** |
| Precision | **80.0%** |
| Recall | **70.0%** |

The model is precision-favoring — detections it makes are usually correct, though recall still leaves room for improvement, which matters most for a safety-relevant navigation aid.

### System-Level Test Results

| Requirement Category | Result |
|---|---|
| OCR (text reading) | ✅ Verified (6/6) |
| Voice command pipeline | ✅ Verified (8/8) |
| YOLO object detection | ✅ Verified (7/7) |
| SLAM mapping & localization | ✅ Verified (5/5) |
| Non-functional requirements (latency, accuracy, noise robustness, resource usage, frame-rate stability) | ✅ Verified (12/12) |

## Comparison with Existing Tools

| Feature | Cane / Guide Dog | Smart Glasses (e.g. Meta) | AVANTI |
|---|---|---|---|
| Obstacle Detection | Basic, short range | Limited | Advanced (Depth + YOLO) |
| Indoor Navigation | ❌ | ❌ | ✅ SLAM + path planning |
| Text Reading (OCR) | ❌ | Basic | ✅ EasyOCR + TTS |
| Voice Interaction | ❌ | Partial | ✅ Full AI assistant |
| Haptic Feedback | Cane vibration only | ❌ | ✅ Programmable vibration |
| Cost | Low–Moderate | Very high | Low-cost hardware |
| Integration | Single function | Partial | Fully integrated |

## Project Status

Functional and non-functional testing has been completed across OCR, voice command handling, YOLO detection, and SLAM subsystems, with all evaluated requirements verified. Ongoing work focuses on expanding the custom obstacle dataset and moving toward offline inference.

## Roadmap

- [ ] Expand the custom obstacle dataset to close the mAP@50 → mAP@50–95 gap
- [ ] Offline / edge AI inference to reduce internet dependency
- [ ] More compact, ergonomic wearable hardware packaging
- [ ] Battery and power-efficiency optimization
- [ ] AI-agent task automation (grocery ordering, ride booking, smartphone control)
- [ ] Structured usability testing with visually impaired participants
- [ ] Smart-home / IoT integration

## Team

| Role | Name |
|---|---|
| Developer | Saif Al Sakit |
| Developer | Ammar Sadeh |
| Supervisor | Prof. Mohammad Abbadi |

Faculty of Information Technology, American University of Madaba — 2026

## References

Key sources informing this project include the WHO *World Report on Vision*, the *Lancet Global Health* blindness projections, and prior work on AI-based navigation aids, smart canes, and smart glasses for visually impaired users. Full citations are available in the accompanying project report.

## License

This project was developed as an academic graduation project at the American University of Madaba. Contact the authors for reuse or collaboration inquiries.
