import { createRenderer, sanitizeMarkup, renderedNodesForRange, sourceNodesForRange, moveRenderedNodes, addPageDivider, renderSource } from "./rendering.js";
import { createPageImage, createLegend } from "./layout.js";
import { PageSync, initSplitter } from "./page-sync.js";
import { downloadLayoutImage } from "./layout-export.js";
import { readModelOptions, readParsingOptions, resetStatus, startStatus, updateStatusPages, finishStatus } from "./config.js";

const $ = (id) => document.getElementById(id);
const fileViewport = $("file-viewport");
const resultViewports = [$("rendered-viewport"), $("source-viewport"), $("layout-viewport")];
const renderer = createRenderer();
let pages = [];
let markdown = "";
let fileName = "document";
let lastFile = null;
let activeTab = "rendered";
let selectedKey = null;
let requestController = null;
let generation = 0;
let imageSignature = "";
let resultReady = false;
let isLoading = false;
let dragDepth = 0;
const blockMap = new Map();
const sync = new PageSync(fileViewport, resultViewports, (page) => {
    $("page-count").textContent = pages.length ? `${page + 1} / ${pages.length}` : "";
});
initSplitter($("document-panels"), $("markdown-splitter"));

function setProgress(stage) {
    isLoading = Boolean(stage);
    if (isLoading) clearFileDrag();
    $("app-container").inert = isLoading;
    $("app-container").setAttribute("aria-busy", String(isLoading));
    $("parse-progress").hidden = !stage;
    $("upload-step").className = `progress-step ${stage === "upload" ? "active" : "complete"}`;
    $("upload-progress-label").textContent = stage === "upload" ? "Uploading..." : "Uploaded.";
    $("parse-step").className = `progress-step${stage === "parse" ? " active" : ""}`;
}

async function request(url, options = {}) {
    const response = await fetch(url, { ...options, signal: requestController.signal });
    let body;
    try { body = await response.json(); } catch { throw new Error(`Server returned an invalid response (${response.status}).`); }
    if (!response.ok || body.code !== 0) {
        const detail = body.detail || body.message || body.error;
        throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status}).`);
    }
    return body.data;
}

function pause(ms, signal) {
    return new Promise((resolve, reject) => {
        const abort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); };
        const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, ms);
        if (signal.aborted) abort(); else signal.addEventListener("abort", abort, { once: true });
    });
}

function pageSection(page, className) {
    const section = document.createElement("section");
    section.className = className;
    section.dataset.pageIndex = String(page.page_num);
    section.setAttribute("aria-label", `Page ${page.page_num + 1}`);
    return section;
}

function appendImagePage(parent, page, mode) {
    const section = pageSection(page, `${mode}-page`);
    const label = document.createElement("div");
    label.className = "page-label";
    label.textContent = `PAGE ${page.page_num + 1}`;
    const image = createPageImage(page, mode, (key) => selectBlock(key, mode), () => selectedKey);
    section.append(label);
    if (mode === "layout") {
        const documentName = fileName;
        const download = document.createElement("button");
        download.type = "button";
        download.className = "page-download";
        download.setAttribute("aria-label", `Download layout for page ${page.page_num + 1}`);
        download.title = "Download page layout as PNG";
        download.innerHTML = $("download-button").innerHTML;
        const errorNotice = document.createElement("div");
        errorNotice.className = "page-download-error";
        errorNotice.setAttribute("role", "alert");
        errorNotice.hidden = true;
        download.addEventListener("click", async (event) => {
            event.stopPropagation();
            download.disabled = true;
            download.setAttribute("aria-busy", "true");
            errorNotice.hidden = true;
            try {
                await downloadLayoutImage(image, documentName, page.page_num);
            } catch (error) {
                errorNotice.textContent = `Download failed: ${error.message || "Please try again."}`;
                errorNotice.hidden = false;
            } finally {
                download.disabled = false;
                download.removeAttribute("aria-busy");
            }
        });
        label.append(download);
        section.append(errorNotice);
    }
    section.append(image);
    if (mode === "file") section.append(createLegend(page.blocks || []));
    parent.append(section);
}

function renderFiles() {
    const fragment = document.createDocumentFragment();
    pages.forEach((page) => appendImagePage(fragment, page, "file"));
    $("file-pages").replaceChildren(fragment);
    $("upload-dropzone").hidden = pages.length > 0;
    sync.setPage(Math.min(sync.page, Math.max(0, pages.length - 1)));
}

function updatePageImages(incoming) {
    const signature = JSON.stringify(incoming.map((page) => [page.page_num, page.image_url, page.page_size]));
    if (signature === imageSignature) return;
    imageSignature = signature;
    pages = incoming.slice().sort((a, b) => a.page_num - b.page_num);
    renderFiles();
}

function showResult(result) {
    markdown = String(result.outputs?.markdown || "");
    const images = new Map(pages.map((page) => [page.page_num, page]));
    pages = (result.demo_pages || []).map((page) => ({ ...images.get(page.page_num), ...page, keep_header_footer: result.metadata?.keep_header_footer === true })).sort((a, b) => a.page_num - b.page_num);
    if (!pages.length) throw new Error("The server returned no document pages.");
    blockMap.clear();
    renderFiles();
    const fragments = resultViewports.map(() => document.createDocumentFragment());
    pages.forEach((page, index) => {
        const rendered = pageSection(page, "rendered-page somarkdown-container theme-dark");
        const source = pageSection(page, "source-page");
        const content = String(page.markdown || "");
        if (content.trim()) {
            try { rendered.append(sanitizeMarkup(renderer.render(content), `page-${page.page_num}-`)); }
            catch { const warning = document.createElement("p"); warning.className = "empty-page"; warning.textContent = "This page could not be rendered. Its Markdown is available in Source."; rendered.append(warning); }
        } else {
            const empty = document.createElement("p"); empty.className = "empty-page"; empty.textContent = "No Markdown content on this page."; rendered.append(empty);
        }
        renderSource(content, source);
        // HTML-table cells are parsed separately by SoMarkDown; their data-line
        // values are cell-local, not page-local. The table root owns this block.
        rendered.querySelectorAll("table[data-line] [data-line]").forEach((node) => node.removeAttribute("data-line"));
        for (const node of rendered.querySelectorAll("[data-line]")) node.classList.add("bi-direction-jump__target");
        const headers = [];
        const footers = [];
        (page.blocks || []).forEach((block, position) => {
            const key = `${page.page_num}:${position}`;
            const sourceNodes = sourceNodesForRange(source, block.start_line, block.end_line);
            let renderedNodes = renderedNodesForRange(rendered, block.start_line, block.end_line);
            const margin = page.keep_header_footer && ["Page-header", "Page-footer"].includes(block.type);
            if (margin) {
                const wrapper = document.createElement("aside");
                wrapper.className = "page-margin bi-direction-jump__target";
                wrapper.dataset.marginKey = key;
                const label = document.createElement("span");
                label.className = "fallback-label";
                label.textContent = block.type === "Page-header" ? "Header" : "Footer";
                const text = document.createElement("div");
                text.className = "page-margin-content";
                // Move the original rendered nodes, keeping their Markdown and
                // page-local line mappings intact without duplicating content.
                moveRenderedNodes(rendered, renderedNodes, text);
                if (!renderedNodes.length && String(block.content || "").trim()) {
                    try {
                        text.append(sanitizeMarkup(renderer.render(String(block.content)), `page-${page.page_num}-margin-${position}-`));
                        text.querySelectorAll("[data-line]").forEach((node) => {
                            if (Number.isInteger(block.start_line)) node.dataset.line = String(Number(node.dataset.line) + block.start_line);
                            else node.removeAttribute("data-line");
                        });
                    } catch { text.textContent = String(block.content); }
                    if (!text.textContent.trim()) text.textContent = String(block.content);
                }
                wrapper.append(label, text);
                (block.type === "Page-header" ? headers : footers).push(wrapper);
                for (const node of sourceNodes) node.dataset.marginKey = key;
                renderedNodes = [wrapper];
            }
            // Markdown definitions (for example unreferenced footnotes) may have no
            // rendered node. Keep their nonempty content visible and selectable.
            if (!renderedNodes.length && sourceNodes.length && String(block.content || "").trim()) {
                const fallback = document.createElement("aside");
                fallback.className = "block-fallback bi-direction-jump__target";
                fallback.dataset.blockKey = key;
                const label = document.createElement("span");
                label.className = "fallback-label";
                label.textContent = String(block.type);
                const text = document.createElement("div");
                text.className = "fallback-content";
                text.textContent = String(block.content);
                fallback.append(label, text);
                rendered.append(fallback);
                renderedNodes = [fallback];
            }
            blockMap.set(key, { page: page.page_num, block, position, rendered: renderedNodes, source: sourceNodes });
        });
        // Footnote fallbacks are appended during the loop; footers follow them.
        rendered.prepend(...headers);
        rendered.append(...footers);
        fragments[0].append(rendered);
        fragments[1].append(source);
        appendImagePage(fragments[2], page, "layout");
        if (index < pages.length - 1) {
            addPageDivider(fragments[0], page.page_num, pages[index + 1].page_num);
            addPageDivider(fragments[1], page.page_num, pages[index + 1].page_num);
        }
    });
    resultViewports.forEach((viewport, index) => viewport.replaceChildren(fragments[index]));
    resultReady = true;
    $("download-button").disabled = false;
    sync.setPage(Math.min(sync.page, pages.length - 1));
}

function clearSelection() {
    selectedKey = null;
    document.querySelectorAll(".bi-direction-jump__target--active, .layout-box.selected").forEach((node) => {
        node.classList.remove("bi-direction-jump__target--active", "selected");
    });
}

function selectBlock(key, origin = "tab") {
    const entry = blockMap.get(key);
    if (!entry) return;
    clearSelection();
    selectedKey = key;
    for (const node of [...entry.rendered, ...entry.source]) node.classList.add("bi-direction-jump__target--active");
    for (const node of document.querySelectorAll(".layout-box")) node.classList.toggle("selected", node.dataset.blockKey === key);
    const viewport = $(`${activeTab}-viewport`);
    const fromResult = ["layout", "rendered", "source"].includes(origin);
    sync.setPage(entry.page, fromResult ? viewport : fileViewport);
    if (fromResult) {
        const fileBox = Array.from(fileViewport.querySelectorAll(".layout-box")).find((node) => node.dataset.blockKey === key);
        sync.reveal(fileViewport, fileBox);
    }
    const target = activeTab === "layout" ? Array.from(viewport.querySelectorAll(".layout-box")).find((node) => node.dataset.blockKey === key) : entry[activeTab][0];
    sync.reveal(viewport, target);

}

function selectTab(name) {
    activeTab = name;
    document.querySelectorAll("[data-tab]").forEach((tab) => {
        const active = tab.dataset.tab === name;
        tab.classList.toggle("active", active);
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
    });
    resultViewports.forEach((viewport) => { viewport.hidden = viewport.id !== `${name}-viewport`; });
    sync.setResult($(`${name}-viewport`));
    if (selectedKey && blockMap.get(selectedKey)?.page === sync.page) selectBlock(selectedKey);
}

async function upload(file) {
    if (!file || isLoading) return;
    resetStatus(file);
    lastFile = null;
    if (!/\.(pdf|png|jpe?g|webp|bmp|tiff?)$/i.test(file.name)) {
        finishStatus("Choose a PDF, PNG, JPEG, WebP, BMP, or TIFF file.", {}, false);
        return;
    }
    if (!file.size) { finishStatus("The selected file is empty.", {}, false); return; }
    lastFile = file;
    let modelOptions;
    try { modelOptions = readModelOptions(); }
    catch (error) { finishStatus(error.message); return; }
    const parsingOptions = readParsingOptions();
    requestController?.abort();
    requestController = new AbortController();
    const signal = requestController.signal;
    const currentGeneration = ++generation;
    fileName = file.name;
    lastFile = file;
    pages = [];
    markdown = "";
    selectedKey = null;
    imageSignature = "";
    resultReady = false;
    blockMap.clear();
    $("download-button").disabled = true;
    $("file-pages").replaceChildren();
    $("upload-dropzone").hidden = false;
    $("page-count").textContent = "";
    resultViewports.forEach((viewport) => {
        const message = document.createElement("div");
        message.className = "empty-state result-empty";
        message.textContent = "Your parsed document will appear here";
        viewport.replaceChildren(message);
    });
    sync.page = 0;
    startStatus();
    setProgress("upload");
    try {
        const form = new FormData();
        form.append("file", file);
        for (const [key, value] of Object.entries(modelOptions)) form.append(key, String(value));
        for (const [key, value] of Object.entries(parsingOptions)) form.append(key, String(value));
        const job = await request("/v1/demo/jobs", { method: "POST", body: form });
        if (currentGeneration !== generation) return;
        setProgress("parse");
        let delay = 650;
        while (currentGeneration === generation) {
            const status = await request(`/v1/demo/jobs/${encodeURIComponent(job.id)}`);
            if (currentGeneration !== generation) return;
            updatePageImages(status.pages || []);
            updateStatusPages(status.pages?.length);
            if (status.status === "failed") throw new Error(typeof status.error === "string" ? status.error : status.error?.message || "Document parsing failed.");
            if (status.status === "success") {
                const result = await request(status.result_url || `/v1/demo/jobs/${encodeURIComponent(job.id)}/result`);
                if (currentGeneration === generation) {
                    showResult(result);
                    setProgress(null);
                    updateStatusPages(pages.length);
                    finishStatus(null, result.metadata);
                }
                return;
            }
            await pause(delay, signal);
            delay = Math.min(1800, delay * 1.15);
        }
    } catch (error) {
        if (error.name === "AbortError" || currentGeneration !== generation) return;
        setProgress(null);
        finishStatus(error.message || "Unable to connect to the server.");
        resultViewports.forEach((viewport) => {
            if (!resultReady) {
                const message = document.createElement("div");
                message.className = "empty-state";
                message.textContent = "Parsing could not finish. Upload a file to try again.";
                viewport.replaceChildren(message);
            }
        });
    }
}

$("retry-button").addEventListener("click", () => upload(lastFile));
$("upload-button").addEventListener("click", () => $("file-input").click());
for (const name of ["rendered", "source"]) {
    $(`${name}-viewport`).addEventListener("click", (event) => {
        if (event.target.closest("a, button, input, select, textarea") || !window.getSelection()?.isCollapsed) return;
        const target = event.target.closest("[data-margin-key]");
        if (target) selectBlock(target.dataset.marginKey, name);
    });
}
for (const viewport of [fileViewport, $("layout-viewport")]) {
    viewport.addEventListener("click", (event) => {
        if (!event.target.closest('.layout-box[role="button"]')) clearSelection();
    });
}
$("upload-dropzone").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => { upload(event.target.files[0]); event.target.value = ""; });
function clearFileDrag() {
    dragDepth = 0;
    $("file-drop-overlay").hidden = true;
}
const hasDraggedFile = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
window.addEventListener("dragenter", (event) => {
    if (!hasDraggedFile(event)) return;
    event.preventDefault();
    if (isLoading) return;
    dragDepth += 1;
    $("file-drop-overlay").hidden = false;
});
window.addEventListener("dragover", (event) => {
    if (!hasDraggedFile(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = isLoading ? "none" : "copy";
    if (!isLoading) $("file-drop-overlay").hidden = false;
});
window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) clearFileDrag();
});
window.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files[0];
    if (hasDraggedFile(event) || file) event.preventDefault();
    clearFileDrag();
    if (!isLoading) upload(file);
});
window.addEventListener("dragend", clearFileDrag);
window.addEventListener("blur", clearFileDrag);
window.addEventListener("keydown", (event) => { if (event.key === "Escape") clearFileDrag(); });
document.querySelectorAll("[data-tab]").forEach((tab) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
        const tabs = Array.from(document.querySelectorAll("[data-tab]"));
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        selectTab(tabs[index].dataset.tab);
        tabs[index].focus();
    });
});
$("download-button").addEventListener("click", () => {
    if (!resultReady) return;
    const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${fileName.replace(/\.[^.]+$/, "") || "document"}.md`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
