(function () {
  "use strict";

  var colors = Object.freeze({
    ink: "#1d1d1f",
    muted: "#6e6e73",
    grid: "#dedee3",
    blue: "#0066cc",
    orange: "#b86b00",
    green: "#00805a",
    red: "#c23832",
    violet: "#8053b5",
    gold: "#957020",
    paper: "#f5f5f7"
  });

  window.AIInfraChartColors = colors;

  if (typeof Chart === "undefined") {
    return;
  }

  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  Chart.defaults.color = colors.muted;
  Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.animation.duration = reducedMotion ? 0 : 500;
  Chart.defaults.layout.padding = 16;
  Chart.defaults.scale.border.display = false;
  Chart.defaults.scale.grid.drawTicks = false;
  Chart.defaults.scale.ticks.padding = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 18;
  Chart.defaults.plugins.tooltip.backgroundColor = colors.ink;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.padding = 12;
}());
