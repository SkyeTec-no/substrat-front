/**
 * Meeting-room hardware: which camera, microphone and speaker Octa uses.
 * Choices persist per browser; "" means the system default. Device labels
 * only appear once the page has camera/mic permission, so the lists are
 * refreshed after the first getUserMedia and whenever devices change.
 */

export const deviceSettings = {
  get cameraId() { return localStorage.getItem("octa-camera") ?? ""; },
  set cameraId(v: string) { localStorage.setItem("octa-camera", v); },
  get micId() { return localStorage.getItem("octa-mic") ?? ""; },
  set micId(v: string) { localStorage.setItem("octa-mic", v); },
  get speakerId() { return localStorage.getItem("octa-speaker") ?? ""; },
  set speakerId(v: string) { localStorage.setItem("octa-speaker", v); },
};

type Kind = "videoinput" | "audioinput" | "audiooutput";

/** Fill a <select> with the devices of one kind, keeping the saved choice. */
export async function fillDeviceSelect(select: HTMLSelectElement, kind: Kind, current: string) {
  let devices: MediaDeviceInfo[] = [];
  try {
    devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === kind);
  } catch {
    // no media devices API (insecure context)
  }
  select.innerHTML = "";
  const def = document.createElement("option");
  def.value = "";
  def.textContent = "system default";
  select.appendChild(def);
  for (const [i, d] of devices.entries()) {
    if (d.deviceId === "default" || d.deviceId === "communications") continue; // aliases of real devices
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `${kind} ${i + 1}`;
    select.appendChild(opt);
  }
  // a saved device that's unplugged still shows, so the choice isn't silently lost
  if (current && ![...select.options].some((o) => o.value === current)) {
    const gone = document.createElement("option");
    gone.value = current;
    gone.textContent = "(saved device not connected)";
    select.appendChild(gone);
  }
  select.value = current;
}

/** Route an AudioContext to the chosen speaker, where the browser allows it. */
export async function routeOutput(ctx: AudioContext | null, speakerId = deviceSettings.speakerId) {
  const c = ctx as (AudioContext & { setSinkId?: (id: string) => Promise<void> }) | null;
  if (!c?.setSinkId) return;
  try {
    await c.setSinkId(speakerId);
  } catch (err) {
    console.warn("could not route audio to the chosen speaker", err);
  }
}
