() => {
  const lines = [];
  for (const element of document.querySelectorAll('h1, h2, h3, p, figcaption')) {
    const vertical = getComputedStyle(element).writingMode.startsWith('vertical');
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    let node, current = [], nodeIndex = -1;
    function finish() {
      while (current.length && !current[0].text.trim()) current.shift();
      while (current.length && !current[current.length-1].text.trim()) current.pop();
      if (current.length) {
        const visible = current.filter(c => c.text.trim());
        lines.push({text: current.map(c => c.text).join(''),
          orientation: vertical ? 'vertical' : 'horizontal',
          element_id: element.dataset.id,
          bbox: [Math.min(...visible.map(c => c.x0)), Math.min(...visible.map(c => c.y0)),
                 Math.max(...visible.map(c => c.x1)), Math.max(...visible.map(c => c.y1))],
          chars: current});
      }
      current = [];
    }
    while ((node = walker.nextNode())) {
      nodeIndex++;
      let offset = 0;
      const combined = node.parentElement.closest('.tcy');
      for (const text of (combined ? [node.textContent] : [...node.textContent])) {
        const range = document.createRange();
        range.setStart(node, offset); offset += text.length; range.setEnd(node, offset);
        // Browser character ranges inside text-combine-upright report the
        // uncombined advances; the inline element has the actual upright box.
        const r = combined ? combined.getBoundingClientRect() : range.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        const c = {text, node_index: nodeIndex, end_offset: offset, x0: r.left + scrollX, y0: r.top + scrollY,
                         x1: r.right + scrollX, y1: r.bottom + scrollY};
        const previous = current.filter(c => c.text.trim()).at(-1);
        if (previous && text.trim()) {
          const cross = vertical ? 'x' : 'y', along = vertical ? 'y' : 'x';
          const size = previous[cross+'1']-previous[cross+'0'];
          const center = a => (a[cross+'0']+a[cross+'1'])/2;
          const gap = c[along+'0']-previous[along+'1'];
          // A wrap changes the cross-axis position; columns remain separate.
          if (Math.abs(center(c)-center(previous)) > size*.45 || gap > size*3 || gap < -size*1.5) finish();
        }
        current.push(c);
      }
    }
    finish();
  }
  return lines;
}
