export class PageSync {
    constructor(fileViewport, resultViewports, onPage) {
        this.fileViewport = fileViewport;
        this.resultViewports = resultViewports;
        this.activeResult = resultViewports[0];
        this.page = 0;
        this.onPage = onPage;
        this.suppressed = new WeakMap();
        for (const viewport of [fileViewport, ...resultViewports]) {
            let pending = false;
            viewport.addEventListener("scroll", () => {
                if (pending) return;
                pending = true;
                requestAnimationFrame(() => {
                    pending = false;
                    if (viewport.hidden || this.suppressed.has(viewport)) return;
                    this.updateFrom(viewport);
                });
            }, { passive: true });
            viewport.addEventListener("load", (event) => {
                if (event.target.tagName === "IMG" && !viewport.hidden && !this.suppressed.has(viewport) && this.currentPage(viewport) !== this.page) this.scrollToPage(viewport, this.page, false);
            }, true);
            for (const event of ["wheel", "touchstart", "pointerdown", "keydown"]) {
                viewport.addEventListener(event, () => this.cancelScroll(viewport), { passive: true });
            }
        }
        // Reflow changes page heights without a scroll event (especially Source
        // wrapping). Preserve the active page when either pane is resized.
        if ("ResizeObserver" in window) {
            const sizes = new WeakMap();
            let resizePending = false;
            this.resizeObserver = new ResizeObserver((entries) => {
                let changed = false;
                for (const entry of entries) {
                    if (entry.target.hidden || !entry.contentRect.width || !entry.contentRect.height) continue;
                    const size = `${entry.contentRect.width}:${entry.contentRect.height}`;
                    if (sizes.has(entry.target) && sizes.get(entry.target) !== size) changed = true;
                    sizes.set(entry.target, size);
                }
                if (!changed || resizePending) return;
                resizePending = true;
                requestAnimationFrame(() => {
                    resizePending = false;
                    this.scrollToPage(this.fileViewport, this.page, false);
                    this.scrollToPage(this.activeResult, this.page, false);
                });
            });
            for (const viewport of [fileViewport, ...resultViewports]) this.resizeObserver.observe(viewport);
        }
    }

    currentPage(viewport) {
        const pages = Array.from(viewport.querySelectorAll("[data-page-index]"));
        if (!pages.length) return null;
        if (viewport.scrollHeight > viewport.clientHeight + 2 && viewport.scrollTop + viewport.clientHeight >= viewport.scrollHeight - 2) {
            return Number(pages[pages.length - 1].dataset.pageIndex);
        }
        const midpoint = viewport.getBoundingClientRect().top + viewport.clientHeight / 2;
        let selected = pages[0];
        let distance = Infinity;
        for (const page of pages) {
            const rect = page.getBoundingClientRect();
            const delta = midpoint < rect.top ? rect.top - midpoint : midpoint > rect.bottom ? midpoint - rect.bottom : 0;
            if (delta < distance) { distance = delta; selected = page; }
        }
        return Number(selected.dataset.pageIndex);
    }

    updateFrom(viewport) {
        const page = this.currentPage(viewport);
        if (page === null || page === this.page) return;
        this.page = page;
        this.onPage(page);
        this.scrollToPage(viewport === this.fileViewport ? this.activeResult : this.fileViewport, page);
    }

    cancelScroll(viewport) {
        const motion = this.suppressed.get(viewport);
        if (!motion) return;
        cancelAnimationFrame(motion.frame);
        this.suppressed.delete(viewport);
    }

    scrollTo(viewport, top, animate = true) {
        const target = Math.max(0, Math.min(top, viewport.scrollHeight - viewport.clientHeight));
        this.cancelScroll(viewport);
        const start = viewport.scrollTop;
        const distance = target - start;
        const motion = { target, frame: 0 };
        this.suppressed.set(viewport, motion);
        // Keep the final scroll event suppressed as well as every intermediate
        // frame. User input cancels this immediately and takes over the pane.
        const finish = () => {
            motion.frame = requestAnimationFrame(() => {
                motion.frame = requestAnimationFrame(() => {
                    if (this.suppressed.get(viewport) === motion) this.suppressed.delete(viewport);
                });
            });
        };
        if (!animate || Math.abs(distance) < 1 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            viewport.scrollTop = target;
            finish();
            return;
        }
        const started = performance.now();
        const duration = Math.min(520, 280 + Math.abs(distance) * 0.12);
        const step = (now) => {
            const progress = Math.min(1, (now - started) / duration);
            const eased = progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
            viewport.scrollTop = start + distance * eased;
            if (progress < 1) motion.frame = requestAnimationFrame(step);
            else finish();
        };
        motion.frame = requestAnimationFrame(step);
    }

    scrollToPage(viewport, index, animate = true) {
        const page = viewport.querySelector(`[data-page-index="${Number(index)}"]`);
        if (!page) return;
        const rect = page.getBoundingClientRect();
        const parent = viewport.getBoundingClientRect();
        const offset = rect.top - parent.top + viewport.scrollTop;
        this.scrollTo(viewport, offset + Math.min(rect.height / 2, viewport.clientHeight / 2) - viewport.clientHeight / 2, animate);
    }

    setPage(page, origin = null) {
        this.page = page;
        this.onPage(page);
        for (const viewport of [this.fileViewport, this.activeResult]) {
            if (viewport !== origin) this.scrollToPage(viewport, page);
        }
    }

    setResult(viewport) {
        if (this.activeResult !== viewport) this.cancelScroll(this.activeResult);
        this.activeResult = viewport;
        this.scrollToPage(viewport, this.page);
    }

    reveal(viewport, node) {
        if (!node) return;
        const rect = node.getBoundingClientRect();
        const parent = viewport.getBoundingClientRect();
        const top = rect.top - parent.top + viewport.scrollTop;
        const bottom = top + rect.height;
        // Selection often follows setPage in the same event. Test against its
        // destination, then retarget the animation instead of jumping twice.
        const destination = this.suppressed.get(viewport)?.target ?? viewport.scrollTop;
        if (top < destination || bottom > destination + viewport.clientHeight) {
            this.scrollTo(viewport, top - 24);
        }
    }
}

export function initSplitter(container, splitter) {
    let ratio = 0.5;
    const apply = (next) => {
        const width = container.clientWidth - splitter.clientWidth;
        if (width <= 0 || getComputedStyle(container).flexDirection === "column") return;
        const min = Math.min(0.45, 280 / width);
        const max = Math.max(0.55, 1 - 320 / width);
        ratio = Math.max(min, Math.min(max, next));
        container.style.setProperty("--file-panel-width", `calc((100% - var(--splitter-hit-size)) * ${ratio})`);
        splitter.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
    };
    splitter.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        splitter.setPointerCapture(event.pointerId);
        container.classList.add("panel-resizing");
    });
    splitter.addEventListener("pointermove", (event) => {
        if (!splitter.hasPointerCapture(event.pointerId)) return;
        apply((event.clientX - container.getBoundingClientRect().left) / (container.clientWidth - splitter.clientWidth));
    });
    for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
        splitter.addEventListener(event, () => container.classList.remove("panel-resizing"));
    }
    splitter.addEventListener("dblclick", () => apply(0.5));
    splitter.addEventListener("keydown", (event) => {
        if (["ArrowLeft", "ArrowRight", "Home", "Enter"].includes(event.key)) {
            event.preventDefault();
            apply(["Home", "Enter"].includes(event.key) ? 0.5 : ratio + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 0.1 : 0.02));
        }
    });
    window.addEventListener("resize", () => apply(ratio));
}
