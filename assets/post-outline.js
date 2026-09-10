(() => {
  const outline = document.querySelector('.post-outline');
  const content = document.querySelector('.post-content');
  if (!outline || !content) return;
  const headings = [...content.querySelectorAll('h2, h3')]
    .filter(heading => !heading.closest('details') && heading.textContent.trim());
  if (headings.length < 2) return;

  const list = outline.querySelector('ol');
  const disclosure = outline.querySelector('details');
  const desktop = window.matchMedia('(min-width: 1200px)');
  let parentItem;
  const links = headings.map((heading, index) => {
    if (!heading.id) {
      let id = `section-${index + 1}`;
      while (document.getElementById(id)) id += '-outline';
      heading.id = id;
    }
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = `#${encodeURIComponent(heading.id)}`;
    link.textContent = heading.textContent.trim();
    item.append(link);
    if (heading.tagName === 'H3' && parentItem) {
      let children = parentItem.querySelector('ol');
      if (!children) {
        children = document.createElement('ol');
        parentItem.append(children);
      }
      children.append(item);
    } else {
      list.append(item);
      if (heading.tagName === 'H2') parentItem = item;
    }
    return link;
  });

  outline.hidden = false;
  document.querySelector('.post-page').classList.add('has-outline');
  const updateMode = () => { disclosure.open = desktop.matches; };
  updateMode();
  desktop.addEventListener('change', updateMode);

  outline.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (!desktop.matches) disclosure.open = false;
    const heading = headings[links.indexOf(link)];
    if (heading) {
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
  });

  let queued = false;
  let current = -1;
  const updateActive = () => {
    let next = -1;
    headings.forEach((heading, index) => {
      if (heading.getBoundingClientRect().top <= 120) next = index;
    });
    if (next !== current) {
      if (current >= 0) links[current].removeAttribute('aria-current');
      if (next >= 0) links[next].setAttribute('aria-current', 'location');
      current = next;
    }
    queued = false;
  };
  window.addEventListener('scroll', () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(updateActive);
    }
  }, { passive: true });
  window.addEventListener('resize', updateActive);
  window.addEventListener('load', updateActive, { once: true });
  updateActive();
})();
