"use strict";

(() => {
  const hero = document.querySelector('.hero');
  const stage = hero.querySelector('.hero-head-comparison');
  const handle = hero.querySelector('.hero-reveal-handle');
  const line = hero.querySelector('.hero-diagonal path');
  const videos = [...hero.querySelectorAll('video')];
  const master = hero.querySelector('[data-hero-after]');
  const followers = videos.filter(video => video !== master);
  const playButton = hero.querySelector('[data-hero-play]');
  const soundButton = hero.querySelector('[data-hero-sound]');
  const status = hero.querySelector('.hero-video-status');
  const icons = {
    play: '<path d="m9 5 11 7-11 7Z"/>',
    pause: '<path d="M8 5v14M16 5v14"/>',
    muted: '<path d="m11 5-6 5H2v4h3l6 5ZM16 9l6 6m0-6-6 6"/>',
    volume: '<path d="m11 5-6 5H2v4h3l6 5ZM16 8a6 6 0 0 1 0 8M19 5a10 10 0 0 1 0 14"/>'
  };
  let reveal = 50;
  let dragging = false;
  let playing = false;
  let starting = false;
  let inView = true;
  let userPaused = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let animation;

  function setReveal(value) {
    reveal = Math.max(0, Math.min(100, value));
    const slant = Math.min(10, reveal, 100 - reveal);
    stage.style.setProperty('--reveal', `${reveal}%`);
    stage.style.setProperty('--reveal-top', `${reveal + slant}%`);
    stage.style.setProperty('--reveal-bottom', `${reveal - slant}%`);
    line.setAttribute('d', `M ${reveal + slant} 0 L ${reveal - slant} 100`);
    handle.setAttribute('aria-valuenow', String(Math.round(reveal)));
    handle.setAttribute('aria-valuetext', `${Math.round(reveal)}% original, ${Math.round(100 - reveal)}% enhanced`);
  }
  function revealAt(event) {
    const box = stage.getBoundingClientRect();
    setReveal((event.clientX - box.left) / box.width * 100 - 20 * (.5 - (event.clientY - box.top) / box.height));
  }
  stage.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    dragging = true;
    stage.setPointerCapture(event.pointerId);
    handle.focus({preventScroll: true});
    revealAt(event);
  });
  stage.addEventListener('pointermove', event => { if (dragging) revealAt(event); });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(type => stage.addEventListener(type, () => { dragging = false; }));
  handle.addEventListener('keydown', event => {
    const increments = {ArrowLeft: -2, ArrowDown: -2, ArrowRight: 2, ArrowUp: 2};
    if (event.key in increments) setReveal(reveal + increments[event.key]);
    else if (event.key === 'Home') setReveal(0);
    else if (event.key === 'End') setReveal(100);
    else return;
    event.preventDefault();
    event.stopPropagation();
  });

  function updatePlayButton() {
    const active = playing || starting;
    playButton.querySelector('svg').innerHTML = icons[active ? 'pause' : 'play'];
    playButton.querySelector('span').textContent = active ? 'Pause' : 'Play';
    playButton.setAttribute('aria-label', `${active ? 'Pause' : 'Play'} background video`);
  }
  function sync() {
    if (!playing) return;
    followers.forEach(video => {
      if (video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime - master.currentTime) > .045) video.currentTime = master.currentTime;
    });
    animation = requestAnimationFrame(sync);
  }
  function pause() {
    playing = false;
    cancelAnimationFrame(animation);
    videos.forEach(video => video.pause());
    updatePlayButton();
  }
  async function play() {
    if (playing || starting || userPaused || !inView || document.hidden) return;
    starting = true;
    updatePlayButton();
    status.textContent = 'Loading synchronized video…';
    document.dispatchEvent(new Event('taomate:hero-play'));
    followers.forEach(video => { if (video.readyState >= 1) video.currentTime = master.currentTime; });
    try {
      await Promise.all(videos.map(video => video.play()));
      if (userPaused || !inView || document.hidden) { pause(); return; }
      followers.forEach(video => { video.currentTime = master.currentTime; });
      playing = true;
      status.textContent = '';
      updatePlayButton();
      sync();
    } catch (error) {
      pause();
      if (error.name === 'NotAllowedError') userPaused = true;
      if (error.name !== 'AbortError') status.textContent = 'Press Play to start the synchronized comparison.';
    } finally {
      starting = false;
      updatePlayButton();
    }
  }
  playButton.addEventListener('click', () => {
    userPaused = playing || starting;
    if (userPaused) pause(); else play();
  });
  soundButton.addEventListener('click', () => {
    master.muted = !master.muted;
    soundButton.querySelector('svg').innerHTML = icons[master.muted ? 'muted' : 'volume'];
    soundButton.querySelector('span').textContent = master.muted ? 'Sound off' : 'Sound on';
    soundButton.setAttribute('aria-pressed', String(!master.muted));
    soundButton.setAttribute('aria-label', `Turn background sound ${master.muted ? 'on' : 'off'}`);
    if (!master.muted) { userPaused = false; play(); }
  });
  videos.forEach(video => {
    video.muted = true;
    video.addEventListener('canplay', () => { if (userPaused) status.textContent = ''; else play(); });
    video.addEventListener('error', () => { pause(); status.textContent = 'Video could not load. Reload the page to try again.'; });
  });
  new IntersectionObserver(entries => {
    inView = entries[0].isIntersecting;
    if (inView) play(); else pause();
  }, {threshold: .12}).observe(hero);
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); else play(); });
  document.addEventListener('play', event => {
    if (event.target instanceof HTMLVideoElement && event.target.closest('.demo-mount')) pause();
  }, true);
  setReveal(50);
  updatePlayButton();
  play();
})();
