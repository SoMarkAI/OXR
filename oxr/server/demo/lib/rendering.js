/* Viewer-compatible rendering and block highlights. No source/preview scroll coupling. */
const HTML_TAGS = new Set("a abbr b blockquote br caption code col colgroup dd del details div dl dt em figcaption figure h1 h2 h3 h4 h5 h6 hr i img ins kbd li mark ol p pre s samp small span strong sub summary sup table tbody td th thead tfoot tr u ul var".split(" "));
const SVG_TAGS = new Set("svg g path line polyline polygon rect circle ellipse text tspan defs clippath mask marker pattern lineargradient radialgradient stop use title desc".split(" "));
const MATH_TAGS = new Set("math semantics annotation mrow mi mo mn ms mtext mspace msup msub msubsup mfrac msqrt mroot mstyle mtable mtr mtd mover munder munderover mpadded mphantom menclose".split(" "));
const DROP_TAGS = new Set("script style iframe frame frameset object embed foreignobject form input button textarea select option link meta base audio video source".split(" "));
const SAFE_ATTRS = new Set("class id title alt width height colspan rowspan scope start reversed type open data-line data-alt aria-hidden aria-label role xmlns display encoding mathvariant viewbox preserveaspectratio d x y x1 y1 x2 y2 dx dy cx cy r rx ry points transform fill fill-opacity fill-rule stroke stroke-width stroke-opacity stroke-linecap stroke-linejoin stroke-dasharray opacity font-size font-family font-weight text-anchor dominant-baseline markerwidth markerheight refx refy orient markerunits offset stop-color stop-opacity gradientunits gradienttransform spreadmethod patternunits patterntransform clippathunits maskunits".split(" "));
const SAFE_STYLES = new Set("color background-color font-size font-family font-weight font-style text-align text-decoration vertical-align display width height min-width max-width min-height max-height margin margin-left margin-right margin-top margin-bottom padding padding-left padding-right padding-top padding-bottom border border-width border-style border-color border-top border-bottom border-left border-right border-radius position top bottom left right line-height letter-spacing white-space transform transform-origin fill fill-opacity stroke stroke-width stroke-opacity opacity overflow".split(" "));

function safeUrl(value, image = false) {
    const compact = value.trim().replace(/[\u0000-\u0020\u007f]/g, "");
    if (image && /^data:image\/(?:png|jpeg|gif|webp|avif|bmp);base64,/i.test(compact)) return value;
    if (compact.startsWith("#")) return value;
    try {
        const url = new URL(compact, window.location.href);
        return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch { return null; }
}

export function sanitizeMarkup(html, prefix) {
    const template = document.createElement("template");
    template.innerHTML = html;
    const visit = (parent) => {
        for (const node of Array.from(parent.children)) {
            const tag = node.localName.toLowerCase();
            if (DROP_TAGS.has(tag)) { node.remove(); continue; }
            if (!HTML_TAGS.has(tag) && !SVG_TAGS.has(tag) && !MATH_TAGS.has(tag)) {
                visit(node);
                node.replaceWith(...node.childNodes);
                continue;
            }
            for (const attr of Array.from(node.attributes)) {
                const name = attr.name.toLowerCase();
                if (name === "style") {
                    const declarations = [];
                    for (const property of Array.from(node.style)) {
                        const value = node.style.getPropertyValue(property);
                        if (SAFE_STYLES.has(property) && !/url\s*\(|expression|@|\\|\/\*|fixed|sticky/i.test(value)) {
                            declarations.push(`${property}:${value}`);
                        }
                    }
                    node.setAttribute("style", declarations.join(";"));
                } else if (name === "href" || name === "xlink:href" || name === "src") {
                    let value = null;
                    if (attr.value.startsWith("#")) value = `#${prefix}${attr.value.slice(1)}`;
                    else if (tag === "a" && name === "href") value = safeUrl(attr.value);
                    else if (tag === "img" && name === "src") value = safeUrl(attr.value, true);
                    if (value === null) node.removeAttribute(attr.name); else node.setAttribute(attr.name, value);
                } else if (["clip-path", "mask", "marker-start", "marker-mid", "marker-end"].includes(name) || (["fill", "stroke"].includes(name) && /url\s*\(/i.test(attr.value))) {
                    const match = /^url\(#([\w:.-]+)\)$/.exec(attr.value);
                    if (match) node.setAttribute(attr.name, `url(#${prefix}${match[1]})`); else node.removeAttribute(attr.name);
                } else if (!SAFE_ATTRS.has(name) || /url\s*\(|javascript|expression/i.test(attr.value)) {
                    node.removeAttribute(attr.name);
                } else if (name === "id") {
                    node.setAttribute("id", prefix + attr.value);
                }
            }
            if (tag === "a" && node.hasAttribute("href") && !node.getAttribute("href").startsWith("#")) {
                node.setAttribute("target", "_blank");
                node.setAttribute("rel", "noopener noreferrer");
            }
            if (tag === "img") { node.loading = "lazy"; node.decoding = "async"; }
            visit(node);
        }
    };
    visit(template.content);
    return template.content;
}

export function createRenderer() {
    return new window.SoMarkDown({
        html: true,
        typographer: true,
        imgDescEnabled: false,
        lineNumbers: { enable: true, nested: true },
        smiles: { disableColors: false },
        toc: { includeLevel: [1, 2, 3] }
    });
}

export function renderedNodesForRange(container, start, end) {
    if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) return [];
    const inRange = (node) => {
        const line = Number(node.getAttribute("data-line"));
        return Number.isInteger(line) && line >= start && line < end;
    };
    // A list container may begin in one OCR block and contain several other blocks.
    // Highlight only whole subtrees whose mapped lines all belong to this block.
    const candidates = Array.from(container.querySelectorAll("[data-line]"))
        .filter((node) => inRange(node) && Array.from(node.querySelectorAll("[data-line]")).every(inRange));
    const candidateSet = new Set(candidates);
    return candidates.filter((node) => {
        for (let parent = node.parentElement; parent && parent !== container; parent = parent.parentElement) {
            if (candidateSet.has(parent)) return false;
        }
        return true;
    });
}

export function sourceNodesForRange(container, start, end) {
    if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) return [];
    return Array.from(container.querySelectorAll("[data-source-line]"))
        .filter((node) => Number(node.dataset.sourceLine) >= start && Number(node.dataset.sourceLine) < end);
}

export function moveRenderedNodes(container, nodes, destination) {
    // Adjacent Markdown lists can share an outer list across OCR blocks. Keep
    // that structural context when extracting only the margin's list items.
    const parents = new Map([[container, destination]]);
    const destinationFor = (parent) => {
        if (parents.has(parent)) return parents.get(parent);
        const clone = parent.cloneNode(false);
        clone.removeAttribute("id");
        clone.removeAttribute("data-line");
        destinationFor(parent.parentElement).append(clone);
        parents.set(parent, clone);
        return clone;
    };
    for (const node of nodes) destinationFor(node.parentElement).append(node);
}

export function addPageDivider(parent, previous, next) {
    const divider = document.createElement("div");
    divider.className = "page-divider";
    divider.setAttribute("role", "separator");
    divider.setAttribute("aria-label", `End of page ${previous + 1}; start of page ${next + 1}`);
    for (const page of [previous, next]) {
        const label = document.createElement("span");
        label.textContent = `PAGE ${page + 1}`;
        divider.append(label);
    }
    parent.append(divider);
}

export function renderSource(markdown, element) {
    const fragment = document.createDocumentFragment();
    String(markdown).split("\n").forEach((text, index) => {
        const line = document.createElement("div");
        line.className = "source-line bi-direction-jump__target";
        line.dataset.sourceLine = String(index);
        const number = document.createElement("span");
        number.className = "source-line-number";
        number.textContent = String(index + 1);
        number.setAttribute("aria-hidden", "true");
        const content = document.createElement("span");
        content.className = "source-line-text";
        content.textContent = text || "\u200b";
        line.append(number, content);
        fragment.append(line);
    });
    element.replaceChildren(fragment);
}
