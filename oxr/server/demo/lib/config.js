const $ = (id) => document.getElementById(id);
const parameters = [
    ["temperature", "Temperature"],
    ["top_p", "Top P"],
    ["repetition_penalty", "Repetition Penalty"],
];
let startedAt = null;
let timer = null;

for (const [key] of parameters) {
    const number = $(`${key}-value`);
    const slider = $(`${key}-slider`);
    slider.addEventListener("input", () => {
        number.value = slider.value;
        number.removeAttribute("aria-invalid");
    });
    number.addEventListener("input", () => {
        const valid = number.value !== "" && number.validity.valid;
        number.setAttribute("aria-invalid", String(!valid));
        if (valid) slider.value = number.value;
    });
}

export function readModelOptions() {
    const options = {};
    for (const [key, label] of parameters) {
        const input = $(`${key}-value`);
        const value = input.valueAsNumber;
        if (!Number.isFinite(value) || !input.validity.valid) {
            input.setAttribute("aria-invalid", "true");
            throw new Error(`${label} must be between ${input.min} and ${input.max}, in steps of 0.01.`);
        }
        options[key] = value;
    }
    return options;
}

function setState(state, label) {
    $("status-state").dataset.state = state;
    $("status-state").textContent = label;
}

function duration(milliseconds) {
    const seconds = Math.max(0, milliseconds) / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)} s`;
    const wholeSeconds = Math.floor(seconds);
    return `${Math.floor(wholeSeconds / 60)}m ${String(wholeSeconds % 60).padStart(2, "0")}s`;
}

function updateElapsed() {
    if (startedAt !== null) $("status-elapsed").textContent = duration(performance.now() - startedAt);
}

export function resetStatus(file) {
    clearInterval(timer);
    timer = null;
    startedAt = null;
    setState("empty", "Empty");
    $("status-file").textContent = file.name;
    $("status-pages").textContent = "—";
    $("status-elapsed").textContent = "—";
    $("status-processing-row").hidden = true;
    $("status-processing").textContent = "—";
    $("status-error").textContent = "";
    $("status-error").hidden = true;
    $("retry-button").hidden = true;
}

export function startStatus() {
    setState("parsing", "Parsing");
    startedAt = performance.now();
    updateElapsed();
    timer = setInterval(updateElapsed, 100);
}

export function updateStatusPages(count) {
    $("status-pages").textContent = Number.isInteger(count) && count > 0 ? String(count) : "—";
}

export function finishStatus(error = null, metadata = {}, retry = true) {
    updateElapsed();
    clearInterval(timer);
    timer = null;
    if (error) {
        setState("error", "Error");
        $("status-error").textContent = error;
        $("status-error").hidden = false;
        $("retry-button").hidden = !retry;
        $("status-section").scrollIntoView({ block: "nearest" });
    } else {
        setState("completed", "Completed");
        if (Number.isInteger(metadata.page_num)) updateStatusPages(metadata.page_num);
        if (Number.isFinite(metadata.processing_time_ms) && metadata.processing_time_ms >= 0) {
            $("status-processing").textContent = duration(metadata.processing_time_ms);
            $("status-processing-row").hidden = false;
        }
    }
}
