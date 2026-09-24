({lines, fraction}) => {
  const content = document.querySelector('.content-body');
  const rect = content.getBoundingClientRect();
  const bodyLines = lines.filter(l => l.element_id !== 'title');
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
  const last = bodyLines[count - 1];
  const element = [...content.querySelectorAll('p')].find(p => p.dataset.id === last.element_id);
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  const char = last.chars.at(-1);
  let node;
  for (let i = 0; i <= char.node_index; i++) node = walker.nextNode();
  const range = document.createRange();
  range.setStart(node, char.end_offset);
  range.setEnd(content, content.childNodes.length);
  range.deleteContents();
  for (const p of content.querySelectorAll('p')) if (!p.textContent.trim()) p.remove();
  return {capacity_lines: capacity, retained_lines: count, target_fraction: fraction,
          content_bbox: [rect.left,rect.top,rect.right,rect.bottom]};
}
