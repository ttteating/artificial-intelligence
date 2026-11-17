/** 极简 markdown 渲染：只覆盖项目里 4 份 md 产物用到的语法（标题/表格/列表/粗体/代码）。 */
function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function inline(s: string): string {
  return esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
}

export function renderMarkdown(md: string): string {
  const lines = (md || '').replace(/\r\n/g, '\n').split('\n')
  const out: string[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) { i++; continue }
    if (/^\s*\|/.test(line)) {
      const rows: string[] = []
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++])
      const cells = rows
        .filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r))
        .map((r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim())))
      if (cells.length) {
        out.push('<div class="md-table"><table class="tbl"><thead><tr>'
          + cells[0].map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>'
          + cells.slice(1).map((r) => '<tr>' + r.map((c) => `<td>${c}</td>`).join('') + '</tr>').join('')
          + '</tbody></table></div>')
      }
      continue
    }
    if (/^```/.test(line)) {
      i++
      const code: string[] = []
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++])
      i++
      out.push(`<pre class="md-pre"><code>${esc(code.join('\n'))}</code></pre>`)
      continue
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line)
    if (h) {
      const level = Math.min(4, h[1].length)
      out.push(`<h${level + 2} class="md-h">${inline(h[2])}</h${level + 2}>`)
      i++
      continue
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ''))}</li>`)
        i++
      }
      out.push(`<ul class="md-list">${items.join('')}</ul>`)
      continue
    }
    if (/^>\s?/.test(line)) {
      const quote: string[] = []
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ''))
      out.push(`<blockquote class="md-quote">${inline(quote.join(' '))}</blockquote>`)
      continue
    }
    const para: string[] = [line]
    i++
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*\||```|>\s?|\s*[-*]\s)/.test(lines[i])) {
      para.push(lines[i++])
    }
    out.push(`<p>${inline(para.join(' '))}</p>`)
  }
  return out.join('\n')
}
