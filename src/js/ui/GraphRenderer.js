export class GraphRenderer {
  constructor(canvas, network) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.network = network;
  }

  setNetwork(network) {
    this.network = network;
  }

  nodeColor(type) {
    if (type === 'input') return getComputedStyle(document.documentElement).getPropertyValue('--input').trim();
    if (type === 'bias') return getComputedStyle(document.documentElement).getPropertyValue('--bias').trim();
    if (type === 'hidden') return getComputedStyle(document.documentElement).getPropertyValue('--hidden').trim();
    return getComputedStyle(document.documentElement).getPropertyValue('--output').trim();
  }

  edgeColor(weight) {
    const alpha = Math.min(0.95, 0.18 + Math.abs(weight) * 0.55);
    return weight >= 0 ? `rgba(125, 211, 252, ${alpha})` : `rgba(255, 123, 123, ${alpha})`;
  }

  layoutNodes() {
    const positions = {};
    const w = this.canvas.width;
    const h = this.canvas.height;
    const columns = new Map();

    for (const node of Object.values(this.network.nodes)) {
      const depth = node.depth;
      if (!columns.has(depth)) columns.set(depth, []);
      columns.get(depth).push(node);
    }

    const depths = [...columns.keys()].sort((a, b) => a - b);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);
    const usableWidth = w - 180;

    for (const depth of depths) {
      const colNodes = columns.get(depth).sort((a, b) => a.id - b.id);
      const ratio = (depth - minDepth) / Math.max(1, maxDepth - minDepth);
      const x = 90 + ratio * usableWidth;
      const total = colNodes.length;
      const top = 110;
      const bottom = h - 90;
      const gap = total === 1 ? 0 : (bottom - top) / (total - 1);

      colNodes.forEach((node, i) => {
        positions[node.id] = { x, y: total === 1 ? (top + bottom) / 2 : top + i * gap };
      });
    }

    return positions;
  }

  drawArrow(x1, y1, x2, y2, color, width, label = null) {
    const { ctx } = this;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const r = 28;
    const sx = x1 + Math.cos(angle) * r;
    const sy = y1 + Math.sin(angle) * r;
    const ex = x2 - Math.cos(angle) * r;
    const ey = y2 - Math.sin(angle) * r;

    ctx.beginPath();
    ctx.moveTo(sx, sy);
    const curve = Math.max(20, Math.abs(x2 - x1) * 0.22);
    ctx.bezierCurveTo(sx + curve, sy, ex - curve, ey, ex, ey);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();

    if (label) {
      ctx.fillStyle = 'rgba(232,236,255,0.9)';
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(label, (sx + ex) / 2, (sy + ey) / 2 - 6);
    }

    const head = 9;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - head * Math.cos(angle - Math.PI / 6), ey - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ex - head * Math.cos(angle + Math.PI / 6), ey - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  drawNode(x, y, node) {
    const { ctx } = this;
    const color = this.nodeColor(node.type);

    ctx.beginPath();
    ctx.arc(x, y, 28, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.stroke();

    ctx.fillStyle = '#0b1020';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.label, x, y);

    ctx.fillStyle = 'rgba(232,236,255,0.85)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText(`d=${node.depth}`, x, y + 42);
  }

  drawLossChart() {
    const { ctx } = this;
    const history = this.network.lossHistory;
    if (history.length < 2) return;

    const x = this.canvas.width - 280;
    const y = 20;
    const w = 240;
    const h = 100;
    const max = Math.max(...history, 0.001);
    const min = Math.min(...history, 0);

    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.strokeRect(x, y, w, h);

    ctx.beginPath();
    history.forEach((v, i) => {
      const px = x + (i / (history.length - 1)) * w;
      const py = y + h - ((v - min) / (max - min + 1e-9)) * h;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });

    ctx.strokeStyle = 'rgba(103,232,165,0.95)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = 'rgba(232,236,255,0.9)';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Loss grafiği', x + 8, y + 16);
  }

  render() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const positions = this.layoutNodes();
    const edges = this.network.getActiveEdges();

    for (const edge of edges) {
      const from = positions[edge.from];
      const to = positions[edge.to];
      if (!from || !to) continue;
      const width = 1 + Math.min(6, Math.abs(edge.weight) * 2.6);
      const label = Math.abs(edge.weight) > 0.7 ? edge.weight.toFixed(2) : null;
      this.drawArrow(from.x, from.y, to.x, to.y, this.edgeColor(edge.weight), width, label);
    }

    for (const node of Object.values(this.network.nodes)) {
      const pos = positions[node.id];
      if (pos) this.drawNode(pos.x, pos.y, node);
    }

    this.drawLossChart();
  }
}
