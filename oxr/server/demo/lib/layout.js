const NS = "http://www.w3.org/2000/svg";
const COLORS = new Map(Object.entries({
    "Text": "#3b82f6", "Title": "#10b981", "Caption": "#ec4899", "Footnote": "#a3a3a3",
    "Formula": "#f59e0b", "Table": "#0ea5e9", "Picture": "#ef4444", "Page-header": "#a3a3a3",
    "Page-footer": "#a3a3a3", "Stamp": "#8b5cf6", "Chemical Structure": "#f97316", "Code Block": "#a3e635"
}));

export function categoryColor(type) {
    if (!COLORS.has(type)) COLORS.set(type, `hsl(${(COLORS.size * 137.508) % 360} 72% 65%)`);
    return COLORS.get(type);
}

function svgElement(tag, attrs) {
    const element = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
    return element;
}

function validBox(block) {
    return Array.isArray(block.bbox) && block.bbox.length === 4 && block.bbox.every(Number.isFinite)
        && block.bbox[2] > block.bbox[0] && block.bbox[3] > block.bbox[1];
}

export function createPageImage(page, mode, onSelect, currentKey) {
    const wrap = document.createElement("div");
    wrap.className = "page-image";
    const w = Number(page.page_size?.w) || 1;
    const h = Number(page.page_size?.h) || 1;
    wrap.style.aspectRatio = `${w} / ${h}`;
    wrap.style.setProperty("--page-ratio", String(w / h));
    const image = document.createElement("img");
    image.alt = `Page ${page.page_num + 1}`;
    image.loading = page.page_num === 0 ? "eager" : "lazy";
    image.decoding = "async";
    image.width = w;
    image.height = h;
    if (page.image_url) image.src = page.image_url;
    image.addEventListener("error", () => {
        const error = document.createElement("span");
        error.className = "image-error";
        error.textContent = "Page image unavailable. Upload the file again to retry.";
        wrap.append(error);
    }, { once: true });
    wrap.append(image);
    const blocks = (page.blocks || []).map((block, position) => ({ ...block, position, key: `${page.page_num}:${position}` })).filter(validBox);
    const readingBlocks = blocks.filter((block) => !["Page-header", "Page-footer"].includes(block.type));
    // Number the same reading sequence as the arrows, independently of JSON IDs.
    const readingNumbers = new Map(readingBlocks.map((block, index) => [block.key, index + 1]));
    const overlay = svgElement("svg", { viewBox: `0 0 ${w} ${h}`, class: "layout-overlay", "aria-label": `Page ${page.page_num + 1} layout` });
    if (mode === "layout" && readingBlocks.length > 1) {
        const markerId = `reading-arrow-${page.page_num}`;
        const defs = svgElement("defs", {});
        const marker = svgElement("marker", { id: markerId, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse", overflow: "visible" });
        marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#000", stroke: "#fff", "stroke-width": 0.7, "stroke-linejoin": "round", "paint-order": "stroke fill" }));
        defs.append(marker);
        overlay.append(defs);
        const outlines = document.createDocumentFragment();
        const lines = document.createDocumentFragment();
        for (let index = 1; index < readingBlocks.length; index += 1) {
            const before = readingBlocks[index - 1].bbox;
            const after = readingBlocks[index].bbox;
            const d = `M ${(before[0] + before[2]) / 2} ${(before[1] + before[3]) / 2} L ${(after[0] + after[2]) / 2} ${(after[1] + after[3]) / 2}`;
            outlines.append(svgElement("path", { d, class: "reading-line-outline" }));
            lines.append(svgElement("path", { d, class: "reading-line", "marker-end": `url(#${markerId})` }));
        }
        overlay.append(outlines, lines);
    }
    const labels = document.createDocumentFragment();
    for (const block of blocks) {
        const [x1, y1, x2, y2] = block.bbox;
        const color = categoryColor(String(block.type));
        const displayOnly = !readingBlocks.includes(block);
        const blockLabel = displayOnly ? String(block.type) : `[${readingNumbers.get(block.key)}] ${block.type}`;
        const rect = svgElement("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, stroke: color, class: `layout-box${displayOnly ? " layout-box--display-only" : ""}`, "aria-label": blockLabel, "data-block-key": block.key });
        rect.style.setProperty("--box-color", color);
        const title = svgElement("title", {});
        title.textContent = displayOnly ? blockLabel : `${blockLabel} — select to highlight`;
        rect.append(title);
        if (!displayOnly) {
            rect.setAttribute("tabindex", "0");
            rect.setAttribute("role", "button");
            rect.addEventListener("click", (event) => {
                // Coincident formula/number boxes remain individually selectable by cycling.
                const bounds = overlay.getBoundingClientRect();
                const x = (event.clientX - bounds.left) / bounds.width * w;
                const y = (event.clientY - bounds.top) / bounds.height * h;
                const hits = readingBlocks.filter((candidate) => x >= candidate.bbox[0] && x <= candidate.bbox[2] && y >= candidate.bbox[1] && y <= candidate.bbox[3]);
                const selected = hits.findIndex((candidate) => candidate.key === currentKey());
                const next = hits.length ? hits[(selected + 1) % hits.length] : block;
                onSelect(next.key, hits.length);
            });
            rect.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(block.key, 1); }
            });
        }
        overlay.append(rect);
        if (mode === "layout") {
            // HTML labels keep a readable CSS-pixel font size while the image and
            // SVG geometry resize. Percent positions stay anchored to the bbox.
            const label = document.createElement("span");
            const left = Math.max(0, Math.min(100, x1 / w * 100));
            label.className = "layout-category-label";
            label.textContent = blockLabel;
            label.setAttribute("aria-hidden", "true");
            label.style.left = `${left}%`;
            label.style.top = `max(0px, calc(${Math.max(0, Math.min(100, y1 / h * 100))}% - var(--layout-label-height, 1.6em)))`;
            label.style.maxWidth = `${100 - left}%`;
            label.style.setProperty("--box-color", color);
            labels.append(label);
        }
    }
    wrap.append(overlay, labels);
    return wrap;
}

export function createLegend(blocks) {
    const legend = document.createElement("div");
    legend.className = "layout-legend";
    for (const type of new Set(blocks.map((block) => String(block.type)))) {
        const item = document.createElement("span");
        item.className = "legend-item";
        const swatch = document.createElement("span");
        swatch.className = "legend-swatch";
        swatch.style.setProperty("--box-color", categoryColor(type));
        item.append(swatch, document.createTextNode(type));
        legend.append(item);
    }
    return legend;
}
