export const EXECUTION_STATE_FIELDS = [
    "studio_id",
    "mesh",
    "export",
    "poses",
    "lights",
    "activeTab",
    "lighting_prompts",
    "animation",
    "background_url",
];

export function poseExecutionState(data) {
    return Object.fromEntries(EXECUTION_STATE_FIELDS.map(name => [name, data?.[name] ?? null]));
}

function canonicalNumber(value) {
    if (!Number.isFinite(value)) throw new Error("Pose execution state contains a non-finite number.");
    if (value === 0) return "0e0";
    const [mantissa, exponent] = value.toExponential(16).split("e");
    return `${mantissa}e${Number(exponent)}`;
}

export function canonicalExecutionValue(value) {
    if (value === null || value === undefined) return "z";
    if (typeof value === "boolean") return value ? "b1" : "b0";
    if (typeof value === "number") return `n${canonicalNumber(value)}`;
    if (typeof value === "string") return `s${JSON.stringify(value)}`;
    if (Array.isArray(value)) return `a[${value.map(canonicalExecutionValue).join(",")}]`;
    if (typeof value === "object") {
        return `o{${Object.keys(value).sort().map(
            key => `${JSON.stringify(key)}:${canonicalExecutionValue(value[key])}`
        ).join(",")}}`;
    }
    throw new Error(`Unsupported pose execution-state value: ${typeof value}`);
}

export function poseExecutionStateToken(data) {
    return canonicalExecutionValue(poseExecutionState(data));
}

export function poseExecutionStatesEqual(left, right) {
    return poseExecutionStateToken(left) === poseExecutionStateToken(right);
}

export async function poseExecutionStateSignature(data) {
    const bytes = new TextEncoder().encode(poseExecutionStateToken(data));
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

export function createStudioId() {
    if (typeof crypto.randomUUID === "function") return `studio-${crypto.randomUUID()}`;
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return `studio-${Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("")}`;
}
