<div align="center">

# OXR FAQ

**English** · [简体中文](FAQ_CN.md)

</div>

## Contents

1. [Product Overview, Capabilities, and Limitations](#product-overview-capabilities-and-limitations)
2. [Deployment and Integration](#deployment-and-integration)
3. [Evaluation and Operating Costs](#evaluation-and-operating-costs)
4. [Model Architecture and Training](#model-architecture-and-training)
5. [Online Services and Licensing](#online-services-and-licensing)
6. [Troubleshooting and Feedback](#troubleshooting-and-feedback)

---

## Product Overview, Capabilities, and Limitations

### 1. What does OXR do, and what is it best suited for?

OXR (Optical Everything Recognition) is an open-source document parsing model from the SoMark team. With 1.3B parameters, it turns PDFs and document images into Agent-Friendly data in Markdown or JSON, with layout coordinates included.

OXR currently performs well on digital documents, including academic papers, research reports, complex tables, formulas, and A3 exam papers. Natural-scene images, such as photos of documents, are not a focus of this version, so we recommend testing thoroughly on your own documents before use.

### 2. What document formats and languages does OXR support?

**File formats:** OXR supports PDFs and common images such as `png`, `jpg`, `jpeg`, `bmp`, `tiff`, and `webp`. `gif`, `heic`, `heif`, and Office files are not supported yet; convert Office files to PDF before submitting. See [API alignment: File formats](https://docs.somark.ai/en/api-reference/oxr/alignment#file-formats).

**Languages:** OXR has mainly been trained on Chinese and English. It should work well on CJK and Western languages, but performance on other languages is not guaranteed. Multilingual support is not a focus of this version, and we will keep improving it in future releases.

### 3. How is OXR different from traditional OCR? What can it output beyond recognized text?

Traditional OCR mainly recognizes text. OXR also recognizes document elements and structure: it can recognize 13 document element types, from formulas and tables to chemical equations, chemical structures, and code blocks, with bounding box coordinates and reading order. See the [README](../README.md) for details.

### 4. Is OXR purely a recognition model, or does it also perform detection?

The OXR model itself does not handle detection. Layout analysis is an independent, pluggable module handled by a separate lightweight model, while OXR handles recognition and reading order.

A built-in layout model is included in the `layout_analysis` folder of the OXR weights, and you can replace it with your own. See [this blog post](https://somark.ai/blogs/the-long-overlooked-performance-cost-in-document-parsing-reading-order) for the design details.

### 5. Can OXR parse CAD files such as DWG and DXF directly? How well does it handle them?

OXR does not parse CAD files such as DWG or DXF directly. Please convert them to PDF or images first.

Because different use cases expect very different results from CAD drawings, OXR parses them as general documents. SoMark's commercial version offers several dedicated solutions for CAD drawings. If you're interested, contact the SoMark team at [somark.ai](https://somark.ai).

### 6. Can OXR directly extract information such as dates, amounts, and IDs into a specified field structure?

Not currently. You can first use OXR to parse the document, then feed the parsed results to a large language model (LLM) to extract the required fields, such as dates, amounts, and ID numbers.

### 7. Does OXR support merging tables across pages?

OXR does not currently support merging tables across pages. If you need this, you can use SoMark.

### 8. What is the relationship between OXR and SoMark? When should I use each one?

OXR is a complete, standalone open-source document parsing model for teams that want to use an open-source model or self-host.

SoMark is a commercial, production-grade document parsing engine from the same team, powered by a separate model. It is not a wrapper around OXR or a "full version" of it. Their APIs are compatible, so you can reuse the same integration tools.

If you need features such as cross-page merging, title level recognition, PDF to Office, or custom development, consider SoMark. See the [OXR overview](https://docs.somark.ai/en/api-reference/oxr).

## Deployment and Integration

### 9. What are the different ways to use OXR?

OXR's API is compatible with the SoMark API, so SoMark's ecosystem fully supports OXR services, including the Free OXR API and self-hosted deployments. You can integrate OXR via:

| Method | Best for |
| --- | --- |
| **API** | Direct HTTP integration |
| **CLI** | Terminals, scripts, and batch jobs |
| **SDK** | Apps, services, and code integration (Python and JavaScript) |
| **Skill** | Skill-based agent platforms |
| **MCP** | MCP-based AI clients |

For the CLI and SDK, just change the Base URL. See the [Getting Started Overview](https://docs.somark.ai/en/documentation/get-started-overview).

### 10. How can OXR integrate a custom layout analysis model? How are cropped regions and coordinates passed to the recognition stage?

OXR supports replacing its layout analysis module. To use your own model:

1. Follow the built-in service's interface and return element categories, bounding boxes, and confidence scores for each page.
2. Configure the service URL in `config.yaml`.

OXR uses the coordinates to crop page regions for content recognition and determines the reading order. If your model already provides reading order, you can configure OXR to skip that step.

For interface and configuration details, see [`services/layout_analyze`](../services/layout_analyze) and [`oxr/model/prompts.py`](../oxr/model/prompts.py) in this repository.

### 11. What hardware do I need to run OXR locally? How much GPU memory is required?

We released the original 1.3B model in BF16. We recommend starting with 24GB of GPU memory on a single GPU, though community reports show it can also run on 12–16GB. See the [README](../README.md) for full requirements.

### 12. Does OXR only run on NVIDIA GPUs? Does it support AMD GPUs, Ascend, and PPU?

No. OXR runs on vLLM, so in principle it can run on any hardware that vLLM supports.

### 13. Can I run OXR with integrated graphics or on a CPU without a dedicated GPU?

Yes, though it will be slower. You can also consider quantized community versions, such as GGUF or ONNX.

### 14. Can OXR run on a Mac?

Yes, but setting up the official version on a Mac takes more effort. You can check whether the community has a solution, and PRs are welcome.

### 15. Is there an on-device version of OXR? Can it run on phones or embedded devices?

No. The OXR series is not designed for on-device use, and we believe on-device scenarios are better served by other approaches.

### 16. Can OXR run completely offline? Does it need to call any external services during parsing?

Yes, it can run fully offline. OXR's weights are open source. Once the runtime environment is set up, it does not call any external services during parsing.

### 17. Can I download and run OXR through Ollama?

Ollama is not officially supported yet. You can check whether the community has a solution, and PRs are welcome.

### 18. Are OXR models available in GGUF or ONNX format?

We do not provide official versions in these formats. You can check whether the community has published one.

### 19. Which versions of vLLM does OXR support? Which version is recommended?

We currently support and recommend **vLLM 0.25.1**. vLLM updates frequently, and we have not tested every version. Since OXR does not use particularly complex operators, later versions should generally work as well.

## Evaluation and Operating Costs

### 20. Where can I find OXR's evaluation results? Are results available for individual languages?

See the [OXR evaluation documentation](benchmark.md), which lists the results, evaluation scope, and related notes for olmOCR-bench, OmniDocBench, and DocParsingBench. Per-language evaluation results are not currently available.

### 21. How can I estimate the per-page parsing cost of a self-hosted OXR deployment?

You can estimate it based on your hardware cost and actual throughput:

> **Per-page cost ≈ hourly hardware cost ÷ pages processed per hour**

Throughput varies considerably with hardware, document types, and concurrency settings, so we recommend benchmarking with your own documents on your target hardware. For reference, OXR reaches about **0.99 pages/s** end-to-end on a single NVIDIA RTX PRO 6000 Blackwell GPU. See the [OXR evaluation documentation](benchmark.md) for the test conditions.

## Model Architecture and Training

### 22. What is the underlying model behind OXR? Was it developed in-house?

Yes. OXR uses an in-house model architecture, pretrained from scratch and built specifically for document parsing.

### 23. What is RSI? Does OXR automatically learn while users are using it?

RSI (Recursive Self-Improvement) is the training approach OXR uses during post-training. See [this blog post](https://somark.ai/blogs/errors-as-data-how-rsi-helps-oxr-achieve-sota-accuracy-and-speed-in-document-parsing) for details.

OXR does not learn from your documents while you use it. We improve the model using online data from the OXR Free API only. SoMark data is strictly confidential and is never used for training.

### 24. Is OXR's RSI training code open source? Where can I find it?

The RSI training code is not open source.

## Online Services and Licensing

### 25. Can I use the same account on the China and international websites?

No. Accounts on SoMark's China site ([somark.cn](https://somark.cn)) and international site ([somark.ai](https://somark.ai)) are separate, so you need to register on each one.

### 26. Can I use OXR commercially? What license terms apply?

Yes. Based on community feedback, we simplified the license on November 6, 2026, by removing the previous competition clause and applying a single size-based rule instead. OXR is released under the OXR Open Source License, a custom license based on Apache 2.0 with additional terms. The key terms are:

- If you and your Affiliates exceed **USD 2,000,000** in either Total Funding (equity or debt) or Annual Revenue, you need a separate commercial license from us, for both internal use and products or services offered to third parties. Personal use and non-commercial research, including academic research, benchmarking, evaluation, and teaching, are not restricted by this term.
- If you use OXR to provide products or services to third parties without a commercial license, you must prominently display **"Powered by SoMark OXR"** with a link to [GitHub](https://github.com/SoMarkAI/OXR).

See the full [LICENSE](../LICENSE) for details. For commercial licensing, contact [official@somark.ai](mailto:official@somark.ai).

## Troubleshooting and Feedback

### 27. Why can reading order results differ between the online demo and a local deployment? How can I troubleshoot the difference?

Differences in GPU, CUDA version, and vLLM version can all lead to different outputs. We recommend first matching the official environment exactly (see the requirements and Docker setup in the [README](../README.md)), then comparing again to rule out environmental factors.

### 28. How do I report an issue? What if I can't share my files publicly?

You can report issues on [GitHub Issues](https://github.com/SoMarkAI/OXR/issues). Please include a description of the problem, steps to reproduce, your environment, and sample files.

If your files can't be shared publicly, submit them through our [feedback page](https://free-oxr.somark.ai/feedback). Please remove any sensitive information first. Our focus is on the parsing issue rather than the specific content of your documents, but please make sure you have the right to share the materials you submit.
