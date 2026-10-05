<div align="center">

<img src="assets/header.png" />

# SoMark｜OXR

[English](README.md) · **简体中文**

精度与速度双 SOTA 的开源文档解析模型

---

让文档成为 Agent-Friendly 数据。

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-OXR--1.0-F59E0B?style=flat-square)](https://huggingface.co/SoMarkAI/OXR-1.0)
[![ModelScope Model](https://img.shields.io/badge/ModelScope-OXR--1.0-624AFF?style=flat-square)](https://modelscope.cn/models/SoMark/OXR-1.0)

</div>

OXR（Optical Everything Recognition）是 SoMark 团队开源的SOTA文档解析模型，参数量1.3B，可将 PDF 和文档图片解析为 Agent 友好数据（Markdown / JSON），并提供版面分析结果。

在架构设计上，OXR 首次将版面分析模块设计为独立、可插拔模块。在提升解析效率的同时，也让业务适配更加灵活。用户可根据业务对文档结构的要求，用自有模型替换内置的版面分析模型，以最小成本实现多场景适配。

---

> 如果这个项目对你有帮助，请点击右上角 ⭐ Star 支持一下，这是对开发者最大的鼓励！


## 许可证

OXR 采用基于 Apache 2.0 的自定义许可证，附带额外条款。

如果您需要更进一步的服务，可联系SoMark团队 → [somark.ai](https://somark.ai/)
