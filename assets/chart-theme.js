(function () {
  "use strict";

  var colors = Object.freeze({
    ink: "#344150",
    muted: "#607080",
    grid: "#dfe4ea",
    blue: "#0072b2",
    orange: "#e69f00",
    green: "#009e73",
    red: "#c43c39",
    violet: "#7a55a3",
    gold: "#b57a24",
    paper: "#fbfaf7"
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
}());
