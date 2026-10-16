/* Keyboard shortcuts apply only while this video has focus. */
(() => {
  const video = document.getElementById('watermark-demo-video');
  if (!video) return;
  video.addEventListener('keydown', (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const key = event.key.toLowerCase();
    const seek = (time) => {
      if (Number.isFinite(video.duration)) {
        video.currentTime = Math.max(0, Math.min(video.duration, time));
      }
    };
    switch (key) {
      case ' ':
      case 'k':
        if (video.paused) video.play().catch(() => {});
        else video.pause();
        break;
      case 'arrowleft': seek(video.currentTime - 5); break;
      case 'arrowright': seek(video.currentTime + 5); break;
      case 'j': seek(video.currentTime - 10); break;
      case 'l': seek(video.currentTime + 10); break;
      case 'home': seek(0); break;
      case 'end': seek(video.duration); break;
      case 'm': video.muted = !video.muted; break;
      default: return;
    }
    event.preventDefault();
  }, { capture: true });
})();
