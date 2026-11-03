"use strict";

const ICONS = {
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7Z" fill="currentColor" stroke="none"/></svg>',
  pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14" stroke-width="3"/></svg>',
  volume: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m11 5-6 5H2v4h3l6 5ZM16 8a6 6 0 0 1 0 8M19 5a10 10 0 0 1 0 14"/></svg>',
  muted: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m11 5-6 5H2v4h3l6 5ZM16 9l6 6m0-6-6 6"/></svg>',
  back: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14m13-14L9 12l10 7Z"/></svg>',
  forward: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M5 5l10 7L5 19Z"/></svg>',
  full: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/></svg>',
  detail: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="15" rx="2"/><rect x="10" y="6" width="6" height="6"/><path d="m13 12 5 8"/></svg>'
};

const escapeHTML = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const clock = value => `${Math.floor(Math.max(0, value) / 60)}:${String(Math.floor(Math.max(0, value) % 60)).padStart(2, "0")}`;
const controllers = [];
let activeController;
const outputLabel = item => item.outputLabel || (item.referenceVariant === "original" ? "TaoMate-FVSR" : "TaoMate-FVSR+");
const referenceLabel = item => item.referenceVariant === "original" ? "Original reference" : "GPT-Image reference";
const referenceImage = item => item.reference;
const mediaVideoUrl = path => String(path || "").startsWith("assets/media/") && String(path).toLowerCase().endsWith(".mp4")
  ? `https://raw.githubusercontent.com/TaoLiveAIGC/TaoMate-FVSR/media/${path}`
  : path;

class ComparisonPlayer {
  constructor(root, cases, fullscene = false) {
    this.root = root;
    this.cases = cases;
    this.fullscene = fullscene;
    this.muted = false;
    this.speed = 1;
    this.playing = false;
    this.generation = 0;
    this.methodSelection = [];
    this.contextRestored = false;
    this.pendingMedia = new WeakMap();
    this.select(0);
    controllers.push(this);
    this.root.addEventListener("pointerdown", () => { activeController = this; });
    this.root.addEventListener("focusin", () => { activeController = this; });
  }

  video(path, poster, label, extra = "") {
    return `<video src="${escapeHTML(mediaVideoUrl(path))}" poster="${escapeHTML(poster || "")}" preload="metadata" playsinline muted aria-label="${escapeHTML(label)}" ${extra}></video>`;
  }

  referencePreview(item) {
    if (!referenceImage(item)) return "";
    return `<div class="head-card reference-card"><div class="media-surface"><button class="reference-preview" data-reference-open aria-haspopup="dialog" aria-label="View ${escapeHTML(item.title)} reference frame"><img src="${escapeHTML(referenceImage(item))}" alt="${escapeHTML(item.title)} reference frame" width="512" height="512"></button></div><div class="head-label"><strong>Reference frame</strong><small>Click to enlarge</small></div></div>`;
  }

  referenceDialog(item) {
    if (!referenceImage(item)) return "";
    return `<dialog class="reference-dialog" aria-labelledby="${this.root.id}-reference-title"><div class="reference-dialog-header"><div><p>Reference frame</p><h2 id="${this.root.id}-reference-title">${escapeHTML(item.title)}</h2></div><button class="icon-button" data-reference-close aria-label="Close reference frame"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg></button></div><img src="${escapeHTML(referenceImage(item))}" alt="${escapeHTML(item.title)} appearance reference" width="512" height="512"><p class="reference-dialog-caption">${escapeHTML(referenceLabel(item))} <span aria-hidden="true">·</span> ${escapeHTML(outputLabel(item))}</p></dialog>`;
  }

  select(index, {time = 0, resume = this.playing} = {}) {
    cancelAnimationFrame(this.animation);
    this.videos?.forEach(video => video.pause());
    this.playing = false;
    this.generation += 1;
    this.current = index;
    this.case = this.cases[index];
    this.requestedTime = Math.min(time, this.case.duration - 1 / (this.case.fps || 25));
    const item = this.case;
    const tabs = this.cases.map((c, i) => `<button class="case-tab" role="tab" aria-selected="${i === index}" data-case="${i}" id="${this.root.id}-tab-${i}" aria-controls="${this.root.id}-panel" tabindex="${i === index ? 0 : -1}"><img src="${escapeHTML(c.sourcePoster)}" alt="" loading="lazy"><span><strong>${escapeHTML(c.title)}</strong><small>${escapeHTML(c.subtitle || (c.task === "dubbing" ? "Replacement audio" : "Original audio"))}</small></span></button>`).join("");
    const label = item.task === "dubbing" ? "Target audio" : "Original audio";
    const name = item.task === "dubbing" ? "Restore + dub" : "Restore facial detail";
    this.root.innerHTML = `<div class="case-navigation"><span class="case-total">${this.cases.length} examples <span aria-hidden="true">·</span> choose a scene</span><div class="case-tabs" role="tablist" aria-label="${this.fullscene ? "Full-scene" : item.task} examples">${tabs}</div></div>
      <div id="${this.root.id}-panel" role="tabpanel" aria-labelledby="${this.root.id}-tab-${index}">
      <div class="viewer-heading"><div><h3>${escapeHTML(item.title)} <span aria-hidden="true">/</span> ${this.fullscene ? "Scene & facial detail" : name}</h3><p>${escapeHTML(item.description || (item.task === "dubbing" ? "Same source video and target audio across outputs" : "Native AIGC source · synchronized facial-detail comparison"))}</p></div>${this.fullscene ? `<div class="view-switch" role="group" aria-label="Comparison view"><button data-view="scene" aria-pressed="${!this.detailMode}">Full scene</button><button data-view="detail" aria-pressed="${!!this.detailMode}">Head detail</button></div>` : `<span class="audio-badge">${ICONS.volume}${label}</span>`}</div>
      ${this.fullscene ? `<div data-comparison-view="scene" ${this.detailMode ? "hidden" : ""}>${this.sceneLayout(item)}<div class="comparison-layout reference-layout"><div class="head-grid"></div></div></div><div data-comparison-view="detail" ${this.detailMode ? "" : "hidden"}>${this.pairedLayout(item, false)}</div>` : this.headLayout(item)}
      <div class="player-controls"><button class="icon-button primary-control" data-action="play" aria-label="Play comparison">${ICONS.play}</button><div class="frame-controls"><button data-action="back" aria-label="Previous frame" title="Previous frame">${ICONS.back}</button><button data-action="forward" aria-label="Next frame" title="Next frame">${ICONS.forward}</button></div><span class="time-label" data-time>0:00 / ${clock(item.duration)}</span><input class="timeline" type="range" min="0" max="${item.duration}" step="${1 / (item.fps || 25)}" value="0" aria-label="${escapeHTML(item.title)} playback position"><select class="speed-select" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" ${this.speed === 1 ? "selected" : ""}>1×</option></select><button class="icon-button" data-action="audio" aria-label="Unmute audio" title="Unmute audio">${ICONS.muted}</button><button class="icon-button fullscreen-button" data-action="fullscreen" aria-label="Fullscreen comparison">${ICONS.full}</button></div><p class="control-status" role="status"></p></div>`;
    this.root.querySelectorAll(".head-grid").forEach(grid => {
      grid.insertAdjacentHTML("beforeend", this.referencePreview(item));
      const note = grid.querySelector(".paired-note");
      if (note) grid.append(note);
    });
    this.root.insertAdjacentHTML("beforeend", this.referenceDialog(item));
    const referenceDialog = this.root.querySelector(".reference-dialog");
    if (referenceDialog) {
      this.root.querySelectorAll("[data-reference-open]").forEach(button => button.addEventListener("click", () => referenceDialog.showModal()));
      this.root.querySelector("[data-reference-close]").addEventListener("click", () => referenceDialog.close());
      referenceDialog.addEventListener("click", event => {
        if (event.target === referenceDialog) referenceDialog.close();
      });
    }
    this.videos = [...this.root.querySelectorAll("video")];
    this.master = this.root.querySelector("video[data-master]");
    this.timeline = this.root.querySelector(".timeline");
    this.timeText = this.root.querySelector("[data-time]");
    this.playButton = this.root.querySelector('.primary-control[data-action="play"]');
    this.audioButton = this.root.querySelector('[data-action="audio"]');
    this.videos.forEach(video => {
      video.muted = true;
      video.playbackRate = this.speed;
      video.addEventListener("loadedmetadata", () => {
        if (!this.videos.includes(video)) return;
        const time = video !== this.master && this.master.readyState >= 1 ? this.master.currentTime : this.requestedTime;
        video.currentTime = Math.min(time, Math.max(0, video.duration - 1 / (this.case.fps || 25)));
      });
      video.addEventListener("error", () => this.status("This video could not be loaded. Check that the media files are present."));
    });
    this.master.muted = this.muted;
    this.master.addEventListener("timeupdate", () => this.updateTime());
    this.sceneVideo = this.videos.find(video => video !== this.master);
    this.sceneVideo?.addEventListener("seeked", () => this.updateROI());
    const currentMaster = this.master;
    this.master.addEventListener("ended", () => {
      if (currentMaster !== this.master) return;
      this.pause(); this.seek(0);
    });
    this.root.querySelectorAll("[data-case]").forEach(button => {
      button.addEventListener("click", () => this.select(Number(button.dataset.case)));
      button.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let next = event.key === "Home" ? 0 : event.key === "End" ? this.cases.length - 1 : (this.current + (event.key === "ArrowRight" ? 1 : -1) + this.cases.length) % this.cases.length;
        this.select(next);
        this.root.querySelector(`[data-case="${next}"]`).focus();
      });
    });
    this.root.querySelectorAll('[data-action="play"]').forEach(button => button.addEventListener("click", () => this.toggle()));
    this.root.querySelector('[data-action="audio"]').addEventListener("click", () => this.toggleAudio());
    this.root.querySelector('[data-action="back"]').addEventListener("click", () => this.step(-1));
    this.root.querySelector('[data-action="forward"]').addEventListener("click", () => this.step(1));
    this.root.querySelector('[data-action="fullscreen"]').addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else this.root.requestFullscreen?.();
    });
    this.timeline.addEventListener("input", () => this.seek(Number(this.timeline.value)));
    this.root.querySelector(".speed-select").value = String(this.speed);
    this.root.querySelector(".speed-select").addEventListener("change", event => {
      this.speed = Number(event.target.value);
      this.videos.forEach(video => { video.playbackRate = this.speed; });
    });
    this.root.querySelectorAll("[data-method-slot]").forEach(select => select.addEventListener("change", () => this.changeMethod(select)));
    this.root.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => {
      if (this.detailMode === (button.dataset.view === "detail")) return;
      this.detailMode = button.dataset.view === "detail";
      this.root.querySelectorAll("[data-comparison-view]").forEach(view => {
        view.hidden = view.dataset.comparisonView !== button.dataset.view;
      });
      this.root.querySelectorAll("[data-view]").forEach(view => view.setAttribute("aria-pressed", String(view === button)));
      this.updateROI(this.position());
    }));
    if (!this.fullscene && item.composite) {
      const switcher = document.createElement("div");
      switcher.className = "view-switch";
      switcher.setAttribute("role", "group");
      switcher.setAttribute("aria-label", "Scene context");
      switcher.innerHTML = '<button data-scene="source" aria-pressed="true">Source scene</button><button data-scene="composite" aria-pressed="false">Restored scene</button>';
      this.root.querySelector(".viewer-heading").append(switcher);
      switcher.querySelectorAll("[data-scene]").forEach(button => button.addEventListener("click", () => {
        const restored = button.dataset.scene === "composite";
        if (restored === this.contextRestored) return;
        this.contextRestored = restored;
        switcher.querySelectorAll("button").forEach(other => other.setAttribute("aria-pressed", String(other === button)));
        this.root.querySelector(".scene-panel .media-tag").textContent = restored ? "Restored scene" : "Source scene";
        this.replaceMedia(this.sceneVideo, {
          video: restored ? item.composite : item.source,
          poster: restored ? item.compositePoster : item.sourcePoster,
          label: `${item.title} ${restored ? "restored" : "source"} scene`
        });
      }));
      if (this.contextRestored) {
        switcher.querySelector('[data-scene="source"]').setAttribute("aria-pressed", "false");
        switcher.querySelector('[data-scene="composite"]').setAttribute("aria-pressed", "true");
        this.sceneVideo.src = mediaVideoUrl(item.composite);
        this.sceneVideo.poster = item.compositePoster;
        this.sceneVideo.setAttribute("aria-label", `${item.title} restored scene`);
        this.root.querySelector(".scene-panel .media-tag").textContent = "Restored scene";
      }
    }
    const heading = this.root.querySelector(".viewer-heading");
    const actions = document.createElement("div");
    actions.className = "viewer-actions";
    [...heading.children].slice(1).forEach(element => actions.append(element));
    const baselineCount = (item.methods || []).filter(method => !["input", "ours"].includes(method.id)).length;
    if (baselineCount) actions.insertAdjacentHTML("afterbegin", `<span class="method-count">${baselineCount} baselines · switch live</span>`);
    heading.append(actions);
    this.root.querySelector(".control-status").insertAdjacentHTML("afterend", '<p class="control-hint"><span><kbd>Space</kbd> Play / pause</span><span><kbd>←</kbd><kbd>→</kbd> Step frame</span><span><kbd>M</kbd> Mute</span></p>');
    this.updateAudioUI();
    this.seek(this.requestedTime, {announce: false});
    this.updateROI();
    if (resume) this.play();
  }

  headLayout(item) {
    const baselines = (item.methods || []).filter(method => !["input", "ours"].includes(method.id));
    if(!baselines.length) return this.pairedLayout({...item, inputHead:item.inputHead || item.input, inputHeadPoster:item.inputHeadPoster || item.inputPoster, oursHead:item.oursHead || item.ours, oursHeadPoster:item.oursHeadPoster || item.oursPoster});
    const preferred = item.task === "dubbing" ? ["flashvsr-full-xdub", "sparkvsr-xdub"] : ["gpen", "sparkvsr"];
    const selected = preferred.map((id, index) => baselines.find(m => m.id === this.methodSelection[index]) || baselines.find(m => m.id === id || m.id.endsWith(`--${id}`)) || baselines[index] || baselines[0]);
    this.methodSelection = selected.map(method => method.id);
    const input = item.methods.find(m => m.id === "input") || {video:item.input, poster:item.inputPoster};
    const ours = item.methods.find(m => m.id === "ours") || {video:item.ours, poster:item.oursPoster};
    const option = (method, chosen) => `<option value="${escapeHTML(method.id)}" ${method.id === chosen?.id ? "selected" : ""}>${escapeHTML(method.label)}</option>`;
    const options = chosen => {
      if (item.task !== "dubbing") return baselines.map(method => option(method, chosen)).join("");
      const groups = new Map();
      baselines.forEach(method => {
        const name = method.label.split(" → ")[0];
        if (!groups.has(name)) groups.set(name, []);
        groups.get(name).push(method);
      });
      return [...groups].map(([name, methods]) => `<optgroup label="${escapeHTML(name)} → dubbing">${methods.map(method => option(method, chosen)).join("")}</optgroup>`).join("");
    };
    const compare = (method, slot) => `<div class="head-card" data-slot="${slot}"><div class="media-surface">${this.video(method.video, method.poster, method.label)}</div><div class="head-label"><select data-method-slot="${slot}" aria-label="Comparison method ${slot + 1}">${options(method)}</select></div></div>`;
    const roi = item.roi ? `<span class="roi-box" style="left:${item.roi.x*100}%;top:${item.roi.y*100}%;width:${item.roi.width*100}%;height:${item.roi.height*100}%"></span>` : "";
    return `<div class="comparison-layout"><div class="scene-panel"><div class="media-surface" style="aspect-ratio:${item.sourceWidth || 1920}/${item.sourceHeight || 1088}">${this.video(item.source, item.sourcePoster, `${item.title} source scene`)}<span class="media-tag">Source scene</span>${roi}<button class="scene-play" data-action="play" aria-label="Play ${escapeHTML(item.title)} comparison"><span>${ICONS.play} Play comparison</span></button></div><div class="scene-caption"><strong>Full scene</strong><span>${item.sourceWidth || 1920} × ${item.sourceHeight || 1088}</span></div><div class="context-explainer">${ICONS.detail}<p>${item.task === "dubbing" ? "Listen to the target audio and compare the reconstructed mouth motion. The original source scene remains a visual context reference." : "The highlighted region is shown on the right. Compare facial structure and detail at the same point in time."}</p></div></div><div class="head-grid"><div class="head-card"><div class="media-surface">${this.video(input.video, input.poster, "Input facial crop")}</div><div class="head-label"><strong>Input</strong><small>Source crop</small></div></div>${compare(selected[0],0)}${compare(selected[1],1)}<div class="head-card ours"><div class="media-surface">${this.video(ours.video, ours.poster, `${outputLabel(item)} output`, "data-master")}</div><div class="head-label"><strong>${escapeHTML(outputLabel(item))}</strong><small>Ours</small></div></div></div></div>`;
  }

  sceneLayout(item) {
    const ratio = (item.sourceWidth || 1536) / (item.sourceHeight || 864);
    return `<div class="fullscene-layout" style="--ratio:${ratio}"><div><div class="media-surface">${this.video(item.source, item.sourcePoster, "Original full scene")}<span class="media-tag">Input</span><button class="scene-play" data-action="play" aria-label="Play ${escapeHTML(item.title)} comparison"><span>${ICONS.play} Play comparison</span></button></div><div class="scene-caption">Original scene</div></div><div class="output"><div class="media-surface">${this.video(item.ours, item.oursPoster, "Restored head composited into full scene", "data-master")}<span class="media-tag">${escapeHTML(outputLabel(item))}</span></div><div class="scene-caption">Restored head · ${escapeHTML(referenceLabel(item))}</div></div></div>`;
  }

  pairedLayout(item, master = true) {
    const roi = item.roi ? `<span class="roi-box" style="left:${item.roi.x*100}%;top:${item.roi.y*100}%;width:${item.roi.width*100}%;height:${item.roi.height*100}%"></span>` : "";
    return `<div class="comparison-layout paired-view"><div class="scene-panel"><div class="media-surface" style="--source-ratio:${item.sourceWidth/item.sourceHeight};aspect-ratio:${item.sourceWidth}/${item.sourceHeight}">${this.video(item.source,item.sourcePoster,`${item.title} source scene`)}<span class="media-tag">Source scene</span>${roi}<button class="scene-play" data-action="play" aria-label="Play ${escapeHTML(item.title)} comparison"><span>${ICONS.play} Play comparison</span></button></div><div class="scene-caption"><strong>Full scene</strong><span>${item.sourceWidth} × ${item.sourceHeight}</span></div></div><div class="head-grid paired-head-grid"><div class="head-card"><div class="media-surface">${this.video(item.inputHead,item.inputHeadPoster,"Input facial crop")}</div><div class="head-label"><strong>Input</strong><small>Source crop</small></div></div><div class="head-card ours"><div class="media-surface">${this.video(item.oursHead,item.oursHeadPoster,`${outputLabel(item)} head output`,master ? "data-master" : "")}</div><div class="head-label"><strong>${escapeHTML(outputLabel(item))}</strong><small>Ours</small></div></div><p class="paired-note">Matched-time head crops<br>${item.task === "dubbing" ? "Target audio" : "Original audio"} · ${escapeHTML(referenceLabel(item))}</p></div></div>`;
  }

  changeMethod(select) {
    const method = this.case.methods.find(item => item.id === select.value);
    this.methodSelection[Number(select.dataset.methodSlot)] = method.id;
    const video = this.root.querySelector(`[data-slot="${select.dataset.methodSlot}"] video`);
    this.replaceMedia(video, method);
  }

  position() {
    return this.master?.readyState >= 1 ? this.master.currentTime : this.requestedTime;
  }

  replaceMedia(video, method) {
    const generation = this.generation;
    const token = {};
    this.pendingMedia.set(video, token);
    const current = () => generation === this.generation && this.pendingMedia.get(video) === token;
    video.addEventListener("loadedmetadata", () => {
      if (!current()) return;
      video.currentTime = Math.min(this.position(), Math.max(0, video.duration - 1 / (this.case.fps || 25)));
      video.playbackRate = this.speed;
      if (this.playing) video.play().catch(() => { if (current()) this.status("This view is loading; the comparison keeps playing."); });
    }, {once:true});
    video.src = mediaVideoUrl(method.video);
    video.poster = method.poster || "";
    video.setAttribute("aria-label", method.label);
    video.muted = true;
    video.playbackRate = this.speed;
    video.load();
  }

  async play() {
    for (const controller of controllers) if (controller !== this) controller.pause();
    if (this.playing) return;
    activeController = this;
    const generation = this.generation;
    const videos = [...this.videos];
    const master = this.master;
    this.playing = true;
    this.status("Loading synchronized videos…");
    this.playButton.innerHTML = ICONS.pause;
    this.playButton.setAttribute("aria-label", "Pause comparison");
    this.root.querySelectorAll(".scene-play").forEach(button => { button.hidden = true; });
    const time = this.master.readyState >= 1 ? this.master.currentTime : this.requestedTime;
    this.requestedTime = time;
    videos.forEach(video => { if (video.readyState >= 1 && Math.abs(video.currentTime - time) > .04) video.currentTime = time; });
    try {
      videos.filter(video => video !== master).forEach(video => video.play().catch(error => {
        if (generation === this.generation && error.name !== "AbortError") this.status("This view is loading; the comparison keeps playing.");
      }));
      await master.play();
      if (generation !== this.generation) return;
      this.playing = true;
      this.status("");
      this.playButton.innerHTML = ICONS.pause;
      this.playButton.setAttribute("aria-label", "Pause comparison");
      this.root.querySelectorAll(".scene-play").forEach(button => { button.hidden = true; });
      cancelAnimationFrame(this.animation);
      this.tick();
    } catch (error) {
      if (generation === this.generation) {
        this.pause();
        this.status("Playback was interrupted. Press play to try again.");
      }
    } finally {
      if (generation === this.generation) this.playButton.disabled = false;
    }
  }

  pause() {
    this.generation += 1;
    this.playing = false;
    cancelAnimationFrame(this.animation);
    this.videos?.forEach(video => video.pause());
    if (this.playButton) {
      this.playButton.innerHTML = ICONS.play;
      this.playButton.setAttribute("aria-label", "Play comparison");
      this.playButton.disabled = false;
      this.root.querySelectorAll(".scene-play").forEach(button => { button.hidden = false; });
    }
  }

  toggle() { if (this.playing) this.pause(); else this.play(); }
  seek(value, {announce = true} = {}) {
    const time = Math.max(0, Math.min(value, this.case.duration - 1/(this.case.fps || 25)));
    this.requestedTime = time;
    this.videos.forEach(video => { if (video.readyState >= 1) video.currentTime = Math.min(time, video.duration); });
    this.timeline.value = String(time);
    this.timeText.textContent = `${clock(time)} / ${clock(this.case.duration)}`;
    if (announce) this.status(`Frame ${Math.round(time * (this.case.fps || 25)) + 1} / ${this.case.frames || Math.round(this.case.duration * 25)}`);
    this.updateROI(time);
  }
  step(direction) { this.pause(); this.seek((this.master.readyState >= 1 ? this.master.currentTime : this.requestedTime) + direction/(this.case.fps || 25)); }
  toggleAudio() { this.muted = !this.muted; this.master.muted = this.muted; this.updateAudioUI(); }
  updateAudioUI() {
    this.audioButton.innerHTML = this.muted ? ICONS.muted : ICONS.volume;
    this.audioButton.setAttribute("aria-label", this.muted ? "Unmute audio" : "Mute audio");
    this.audioButton.title = this.muted ? "Unmute audio" : "Mute audio";
  }
  status(message) { const status=this.root.querySelector(".control-status"); if(status) status.textContent=message; }
  updateTime() {
    if (!this.timeline || !this.master) return;
    this.timeline.value = String(this.master.currentTime);
    this.timeText.textContent = `${clock(this.master.currentTime)} / ${clock(this.case.duration)}`;
    this.updateROI();
  }
  updateROI(time) {
    const box = this.root.querySelector(".roi-box");
    const boxes = this.case.roiFrames;
    if (!box || !boxes?.length) return;
    const t = time ?? (this.sceneVideo?.readyState >= 1 ? this.sceneVideo.currentTime : this.requestedTime);
    const roi = boxes[Math.min(boxes.length - 1, Math.max(0, Math.floor(t * (this.case.fps || 25) + .001)))];
    Object.assign(box.style, {left:`${roi.x*100}%`,top:`${roi.y*100}%`,width:`${roi.width*100}%`,height:`${roi.height*100}%`});
  }
  tick() {
    if (!this.playing) return;
    const time=this.master.currentTime;
    this.videos.forEach(video => { if (video !== this.master && video.readyState >= 2 && Math.abs(video.currentTime-time) > .065) video.currentTime=time; });
    this.updateTime();
    this.animation=requestAnimationFrame(() => this.tick());
  }
}

async function loadDemos() {
  const catalog = await fetch("assets/media/catalog.json").then(response => { if(!response.ok) throw new Error("Media catalog unavailable"); return response.json(); });
  const restoration = catalog.cases.filter(item => item.task === "restoration");
  const dubbing = catalog.cases.filter(item => ["dubbing","crossaudio"].includes(item.task)).map(item => ({...item, task:"dubbing"}));
  if(restoration.length) new ComparisonPlayer(document.getElementById("restoration-demo"), restoration);
  if(dubbing.length) new ComparisonPlayer(document.getElementById("dubbing-demo"), dubbing);
  try {
    const full = await fetch("assets/media/fullscene-catalog.json").then(response => {if(!response.ok) throw new Error("Full-scene catalog unavailable"); return response.json();});
    new ComparisonPlayer(document.getElementById("fullscene-demo"), full.cases, true);
  } catch (error) {
    document.getElementById("fullscene-demo").innerHTML='<p class="loading-message">Full-scene media are not present in this copy. See the README for the media folder layout.</p>';
  }
  renderLibrary();
}

function renderLibrary() {
  const entries = controllers.flatMap(controller => controller.cases.map((item,index) => ({item,index,controller,category:controller.fullscene ? "fullscene" : item.task})));
  const names = {restoration:"Facial restoration",dubbing:"Replacement audio",fullscene:"Full scene"};
  const library = document.getElementById("video-library");
  const render = filter => {
    const visible = entries.filter(entry => filter === "all" || entry.category === filter);
    library.innerHTML = visible.map(({item,index,controller,category}) => `<button class="library-card" data-player="${controller.root.id}" data-index="${index}" aria-label="Watch ${escapeHTML(item.title)} ${names[category]} comparison"><div class="library-poster"><img src="${escapeHTML(item.sourcePoster)}" alt="${escapeHTML(item.title)} scene" loading="lazy"><span class="library-play">${ICONS.play}</span><span class="library-duration">${item.duration.toFixed(1)} s</span><span class="library-task">${names[category]}</span></div><div class="library-caption"><strong>${escapeHTML(item.title)}</strong><span>${escapeHTML(outputLabel(item))} <span aria-hidden="true">↗</span></span></div></button>`).join("");
    library.querySelectorAll("[data-player]").forEach(button => button.addEventListener("click", () => {
      const controller = controllers.find(c => c.root.id === button.dataset.player);
      controller.select(Number(button.dataset.index));
      controller.root.scrollIntoView({behavior:"smooth",block:"start"});
      controller.play();
    }));
  };
  document.querySelectorAll("[data-filter]").forEach(button => {
    const filter=button.dataset.filter;
    button.querySelector("[data-count]").textContent=entries.filter(entry=>filter==="all" || entry.category===filter).length;
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-filter]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));
      render(filter);
    });
  });
  document.getElementById("collection-summary").innerHTML=`${entries.length} video comparisons across generated scenes.<br>Choose a thumbnail to play its synchronized comparison.`;
  render("all");
}
loadDemos().catch(error => {
  for (const root of document.querySelectorAll(".demo-mount")) root.innerHTML='<p class="loading-message">Start the local server to load the demo and its media. See README.md for the preview command.</p>';
  console.error(error);
});

document.addEventListener("visibilitychange", () => { if(document.hidden) controllers.forEach(controller => controller.pause()); });
document.addEventListener("taomate:hero-play", () => controllers.forEach(controller => controller.pause()));
document.addEventListener("keydown", event => {
  if (!activeController || event.altKey || event.ctrlKey || event.metaKey) return;
  const target = event.target;
  if (target.closest(".hero, input, select, textarea, dialog, [contenteditable='true'], [role='tab']")) return;
  if (event.code === "Space" && !target.closest("button, a")) {
    event.preventDefault(); activeController.toggle();
  } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault(); activeController.step(event.key === "ArrowRight" ? 1 : -1);
  } else if (event.key.toLowerCase() === "m") {
    event.preventDefault(); activeController.toggleAudio();
  }
});
const sectionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => { if(entry.isIntersecting) {
    document.querySelectorAll(".navigation nav a").forEach(link => link.classList.toggle("active",link.hash === `#${entry.target.id}`));
  }});
}, {rootMargin:"-15% 0px -65% 0px"});
document.querySelectorAll("main section[id]").forEach(section => sectionObserver.observe(section));
