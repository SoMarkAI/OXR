const SVG_NS = "http://www.w3.org/2000/svg";

function imageLoaded(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Unable to render the layout image."));
        image.src = src;
    });
}

function dataURL(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Unable to read the page image."));
        reader.readAsDataURL(blob);
    });
}

function snapshot(wrap, width, height) {
    const clone = wrap.cloneNode(true);
    const originals = [wrap, ...wrap.querySelectorAll("*")];
    const copies = [clone, ...clone.querySelectorAll("*")];
    for (let index = 0; index < originals.length; index += 1) {
        const computed = getComputedStyle(originals[index]);
        const target = copies[index];
        target.removeAttribute("id");
        // Freeze the actual appearance, including selection, label truncation,
        // responsive dimensions and non-scaling SVG strokes, before any await.
        for (const property of computed) {
            let value = computed.getPropertyValue(property);
            // Browser-computed marker URLs may include the current document URL.
            // The standalone snapshot must resolve them within its own SVG.
            value = value.replace(/url\(["']?[^)"']*#([^\s)"']+)["']?\)/g, "url(#$1)");
            target.style.setProperty(property, value);
        }
        target.style.setProperty("animation", "none");
        target.style.setProperty("transition", "none");
        // Marker IDs are referenced by the reading-order paths.
        if (originals[index].namespaceURI === SVG_NS && originals[index].id) {
            target.id = originals[index].id;
        }
    }
    Object.assign(clone.style, {
        position: "relative", width: `${width}px`, height: `${height}px`,
        minWidth: "0", minHeight: "0", maxWidth: "none", maxHeight: "none",
        margin: "0", transform: "none", aspectRatio: "auto"
    });
    const image = clone.querySelector("img");
    image.removeAttribute("srcset");
    image.removeAttribute("sizes");
    image.setAttribute("loading", "eager");
    return clone;
}

/** Download exactly the current page drawing, excluding its surrounding controls. */
export async function downloadLayoutImage(wrap, fileName, pageNum) {
    const image = wrap.querySelector("img");
    const { width, height } = wrap.getBoundingClientRect();
    const source = image?.currentSrc || image?.src;
    if (!source || !width || !height || wrap.querySelector(".image-error")) {
        throw new Error("The page image is not available to download.");
    }
    const clone = snapshot(wrap, width, height);
    const response = await fetch(source, { credentials: "same-origin" });
    if (!response.ok) throw new Error("Unable to load the page image. Please try again.");
    const rasterURL = await dataURL(await response.blob());
    const raster = await imageLoaded(rasterURL);
    clone.querySelector("img").src = rasterURL;

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const content = document.createElementNS(SVG_NS, "foreignObject");
    content.setAttribute("width", "100%");
    content.setAttribute("height", "100%");
    content.append(clone);
    svg.append(content);
    // A self-contained data URL keeps foreignObject canvas rendering origin-clean;
    // a Blob URL SVG can taint the canvas even when all its images are embedded.
    const serialized = new XMLSerializer().serializeToString(svg);
    const drawing = await imageLoaded(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(serialized)}`);
    const scale = Math.max(raster.naturalWidth / width, raster.naturalHeight / height, window.devicePixelRatio || 1);
    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(width * scale);
    canvas.height = Math.ceil(height * scale);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("This browser cannot export the layout image.");
    context.drawImage(drawing, 0, 0, canvas.width, canvas.height);
    const png = await new Promise((resolve, reject) => {
        canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Unable to export the layout image.")), "image/png");
    });
    const url = URL.createObjectURL(png);
    const link = document.createElement("a");
    const baseName = String(fileName || "document").replace(/\.[^.]+$/, "").replace(/[\\/\u0000-\u001f]/g, "_");
    link.download = `${baseName}-page-${pageNum + 1}-layout.png`;
    link.href = url;
    document.body.append(link);
    try { link.click(); } finally {
        link.remove();
        // Give the browser time to consume the download before releasing its URL.
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
}
