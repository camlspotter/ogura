({lines, fraction}) => {
  const content = document.querySelector('.content-body');
  const rect = content.getBoundingClientRect();
  const bodyIds = new Set([...content.querySelectorAll('p, h2, h3')].map(p => p.dataset.id));
  const bodyLines = lines.filter(l => bodyIds.has(l.element_id));
  const fits = l => {
    const [x0,y0,x1,y1] = l.bbox;
    return x0 >= rect.left - .5 && y0 >= rect.top - .5 && x1 <= rect.right + .5 && y1 <= rect.bottom + .5;
  };
  // Keep a contiguous prefix: never retain later fragments after an overflowing line.
  let count = bodyLines.findIndex(l => !fits(l));
  if (count < 0) throw new Error('Not enough source text to overflow the page');
  if (!count) throw new Error('No complete body line fits on the page: ' + JSON.stringify({first: bodyLines[0]?.bbox, content: [rect.left,rect.top,rect.right,rect.bottom]}));
  const capacity = count;
  count = Math.max(1, Math.floor(count * fraction));
  // A retained page must not end with a heading without its following text.
  const headingIds = new Set([...content.querySelectorAll('h2, h3')].map(e => e.dataset.id));
  while (count > 0 && headingIds.has(bodyLines[count - 1].element_id)) count--;
  if (!count) throw new Error('No body text fits after heading');
  const last = bodyLines[count - 1];
  const element = [...content.querySelectorAll('p, h2, h3')].find(p => p.dataset.id === last.element_id);
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  const char = last.chars.at(-1);
  let node;
  for (let i = 0; i <= char.node_index; i++) node = walker.nextNode();
  const range = document.createRange();
  range.setStart(node, char.end_offset);
  range.setEnd(content, content.childNodes.length);
  range.deleteContents();
  for (const p of content.querySelectorAll('p, h2, h3')) if (!p.textContent.trim()) p.remove();
  return {capacity_lines: capacity, retained_lines: count, target_fraction: fraction,
          content_bbox: [rect.left,rect.top,rect.right,rect.bottom]};
}
