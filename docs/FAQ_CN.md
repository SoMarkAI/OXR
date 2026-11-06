<div align="center">

# OXR 常见问题

[English](FAQ.md) · **简体中文**

</div>

## 目录

1. [产品定位与能力边界](#一产品定位与能力边界)
2. [部署环境与接入方式](#二部署环境与接入方式)
3. [效果评估与运行成本](#三效果评估与运行成本)
4. [模型架构与训练原理](#四模型架构与训练原理)
5. [在线服务与许可协议](#五在线服务与许可协议)
6. [问题排查与反馈](#六问题排查与反馈)

---

## 一、产品定位与能力边界

### 1. OXR 是做什么的，适合用在哪些场景？

OXR（Optical Everything Recognition）是 SoMark 团队开源的文档解析模型，参数量 1.3B，可将 PDF 和文档图片解析为 Agent-Friendly 数据（Markdown / JSON），并提供版面分析结果。

目前 OXR 在电子版文档上效果较好，如学术论文、研报、复杂表格、公式和 A3 试卷等；自然场景图片（如拍摄的文档）不是当前版本的重点优化方向，建议在使用前结合自己的文档充分测试评估。

### 2. OXR 支持哪些文档格式和语言？

**文件格式：** OXR 支持 PDF 和 `png`、`jpg`、`jpeg`、`bmp`、`tiff`、`webp` 等常见图片，暂不支持 `gif`、`heic`、`heif` 及 Office 文件，Office 文件请先转换为 PDF 再提交。详见[接口对齐：文件格式](https://docs.somark.cn/api-reference/oxr/alignment#文件格式)。

**语言：** OXR 目前主要针对中英文进行了充分训练，在 CJK 和西文语系上预计效果较好，其他语言暂不保证效果。多语言不是当前版本的重点，后续版本会持续迭代。

### 3. OXR 和传统 OCR 有什么区别？除了识别文字，还能输出什么？

传统 OCR 主要识别文字，OXR 还可识别文档中的多种元素与结构。OXR 可识别公式、表格、化学方程式、化学结构式和代码块等 13 类文档元素，并提供对应的版面坐标，同时输出阅读顺序。详见 [README](../README_CN.md)。

### 4. OXR 是纯粹识别模型吗？有没有检测功能？

OXR 模型本身不负责检测。我们将版面分析设计为独立、可插拔模块，由单独的轻量模型完成，OXR 负责内容识别与阅读顺序判断。

OXR 权重中已附带内置的版面分析模型（`layout_analysis` 文件夹），也可替换为自有模型。设计思路详见[这篇博客](https://somark.ai/blogs/the-long-overlooked-performance-cost-in-document-parsing-reading-order)。

### 5. OXR 支持直接解析 DWG、DXF 等 CAD 文件吗？解析效果如何？

OXR 不支持直接解析 DWG、DXF 等 CAD 文件，建议先转换为 PDF 或图片再提交。

由于不同场景对 CAD 图纸的解析目标差异很大，OXR 会将其作为通用文档进行解析。SoMark 商业版针对 CAD 图纸提供多种专门的处理方案，如有需要，可联系 SoMark 团队：[somark.cn](https://somark.cn)。

### 6. OXR 能否按照指定的字段结构，直接提取日期、金额、编号等信息？

目前 OXR 暂不支持按照指定字段结构直接提取信息。您可以先用 OXR 解析文档，再将解析结果交给后续的大语言模型，提取日期、金额、编号等所需字段。

### 7. OXR 是否支持跨页表格拼接？

OXR 目前不支持跨页表格拼接。需要跨页表格拼接能力时，可以使用 SoMark。

### 8. OXR 和 SoMark 是什么关系？什么情况下适合使用哪一个？

OXR 是完整、可独立使用的开源文档解析模型，适合希望使用开源模型或需要自部署的场景。

SoMark 是同一团队打造的商业化生产级文档解析引擎，由独立模型驱动，并非 OXR 的封装或“完整版”。两者的 API 兼容，可复用调用工具。

如需跨页拼接、标题层级识别、PDF 转 Office 等功能或定制服务，可考虑 SoMark。详见 [OXR 概览](https://docs.somark.cn/api-reference/oxr)。

## 二、部署环境与接入方式

### 9. OXR 有哪些使用方式？

OXR 的 API 与 SoMark API 兼容，SoMark 生态完全支持 OXR 服务（Free OXR API 或自部署），可通过以下方式接入：

| 方式 | 适用场景 |
| --- | --- |
| **API** | 直接调用 HTTP 接口 |
| **CLI** | 终端、脚本和批处理 |
| **SDK** | 接入代码和服务端，支持 Python 和 JS |
| **Skill** | Agent 平台里的 Skill 接入 |
| **MCP** | MCP 客户端里的助手调用 |

使用 CLI 和 SDK 时只需替换 Base URL。详见 [SoMark 入门总览](https://docs.somark.cn/documentation/get-started-overview)。

### 10. OXR 如何接入自定义版面分析模型？区域裁剪和坐标如何传递给后续识别？

OXR 的版面分析模块支持替换。接入自有模型时：

1. 按内置版面分析服务的接口约定，返回每页的元素类别、坐标框和置信度；
2. 在 `config.yaml` 中配置服务地址。

OXR 会按坐标裁剪页面区域，完成内容识别与阅读顺序判断；如果自有模型已提供阅读顺序，可通过配置跳过该步骤。

接口与配置细节可参考本仓库中的 [`services/layout_analyze`](../services/layout_analyze) 和 [`oxr/model/prompts.py`](../oxr/model/prompts.py)。

### 11. 本地部署 OXR 需要什么配置？至少需要多少显存？

我们发布的是 1.3B 参数的 BF16 原版模型，建议单卡显存 24GB 起步。从社区反馈来看，12～16GB 显存也能运行。完整环境要求见 [README](../README_CN.md)。

### 12. OXR 只能在 NVIDIA 显卡上运行吗？AMD、昇腾、PPU 支持吗？

不限于 NVIDIA 显卡。OXR 基于 vLLM 运行，理论上 vLLM 支持的硬件都可以运行 OXR。

### 13. 没有独显，能用集显或纯 CPU 跑 OXR 吗？

可以，只是速度会慢一些。也可以考虑社区的量化版本，例如 GGUF、ONNX 格式。

### 14. OXR 能在 Mac 上跑吗？

可以，但官方版本在 Mac 上部署相对繁琐。可以关注社区是否有相关方案，也欢迎提交 PR。

### 15. OXR 有端侧版本吗？能在手机或嵌入式设备上运行吗？

暂无。OXR 系列并非面向端侧设计，我们认为端侧场景应有更合适的方案。

### 16. OXR 能完全离线运行吗？解析时还需要调用外部服务吗？

可以。OXR 权重已开源，准备好运行环境后即可完全离线运行，解析时无需调用任何外部服务。

### 17. OXR 可以通过 Ollama 下载和运行吗？

官方暂未提供 Ollama 支持，可以关注社区是否有相关方案，也欢迎提交 PR。

### 18. 有 GGUF 或 ONNX 格式的 OXR 模型吗？

官方暂未提供，可以关注社区是否有相关版本。

### 19. OXR 支持哪些版本的 vLLM？推荐用哪个版本？

官方当前适配的版本是 **vLLM 0.25.1**，推荐使用该版本。vLLM 更新较快，我们暂未逐一测试适配其他版本；由于 OXR 没有使用特别复杂的算子，理论上 0.25.1 之后的版本一般也能正常支持。

## 三、效果评估与运行成本

### 20. OXR 的评测结果在哪里查看？是否有按语言分别统计的结果？

可查看 [OXR 评测文档](benchmark.md)，其中列出了 olmOCR-bench、OmniDocBench 和 DocParsingBench 的成绩、评测范围及相关说明。目前暂未提供按语言分别统计的评测结果。

### 21. 自部署 OXR 的单页解析成本如何估算？

可以根据运行设备的成本和实际吞吐速度来估算：

> **单页成本 ≈ 设备每小时成本 ÷ 每小时实际处理页数**

吞吐速度会因硬件、文档类型和并发配置不同而有较大差异，建议在目标设备上用自己的文档实测。作为参考，在单张 NVIDIA RTX PRO 6000 Blackwell GPU 上，OXR 端到端吞吐约为 **0.99 pages/s**，测试条件详见 [OXR 评测文档](benchmark.md)。

## 四、模型架构与训练原理

### 22. OXR 的底层模型是什么？是自研的吗？

是自研的。OXR 采用自研模型结构，从零开始预训练（Pretrained from Scratch），是专用于文档解析任务的模型。

### 23. RSI 是什么？OXR 会在用户使用过程中自动学习吗？

RSI（Recursive Self-Improvement，递归自我改进）是 OXR 在后训练阶段采用的训练方法，详见[这篇博客](https://somark.ai/blogs/errors-as-data-how-rsi-helps-oxr-achieve-sota-accuracy-and-speed-in-document-parsing)。

OXR 不会在使用过程中自动学习。我们会基于 OXR Free API 的线上数据对模型进行迭代优化。特别说明：商业版产品 SoMark 的数据严格保密，不会用于训练。

### 24. OXR 的 RSI 训练代码开源了吗？在哪里能找到？

RSI 训练代码暂未开源。

## 五、在线服务与许可协议

### 25. 官网国内站与海外站的账号是否互通？

不互通。SoMark 国内站（[somark.cn](https://somark.cn)）与海外站（[somark.ai](https://somark.ai)）的账号相互独立，需要分别注册。

### 26. OXR 可以商用吗？需要遵守哪些许可条款？

可以商用。我们根据社区反馈，于 2026 年 11 月 6 日简化了许可条款，取消了此前的竞争性条款，改为统一按规模判断。OXR 采用 OXR Open Source License，这是基于 Apache 2.0 的自定义许可证，附带额外条款，主要包括：

- 您及关联方的累计融资总额（含股权和债务融资）或年度总收入任一超过 **200 万美元**时，无论内部使用还是对外提供，均需先向我们取得单独的商业许可；个人使用及非商业研究（含学术研究、基准测试、评测和教学）不受此限；
- 未取得商业许可而使用 OXR 向第三方提供产品或服务时，需显著标明 **“Powered by SoMark OXR”**，并链接至 [GitHub](https://github.com/SoMarkAI/OXR)。

具体以 [LICENSE](../LICENSE) 原文为准。商业授权请联系 [somark@soulcode.cn](mailto:somark@soulcode.cn)。

## 六、问题排查与反馈

### 27. 为什么在线 Demo 和本地部署的阅读顺序结果不完全一致？该怎么排查？

不同的 GPU、CUDA 版本和 vLLM 版本都可能导致输出结果不同。建议先将环境与官方保持完全一致（参考 [README](../README_CN.md) 中的环境要求和 Docker 部署方式）再进行对比，以排除环境因素。

### 28. 遇到问题如何反馈？资料不便公开时怎么办？

可通过 [GitHub Issues](https://github.com/SoMarkAI/OXR/issues) 反馈，提交时请附上问题描述、复现步骤、运行环境和示例文件。

如资料不便公开，可通过[反馈页面](https://free-oxr.somark.cn/feedback)提交。请提前对资料做脱敏处理，我们不关注文档的具体内容，但需要避免任何版权问题。
