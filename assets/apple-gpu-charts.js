(function () {
  "use strict";

  if (typeof Chart === "undefined") {
    return;
  }

  var colors = window.AIInfraChartColors || {
    ink: "#344150",
    muted: "#607080",
    grid: "#dfe4ea",
    blue: "#0072b2",
    green: "#009e73",
    paper: "#fbfaf7"
  };

  var legacyColor = "#8a96a3";
  var naxColor = colors.green;

  var speedupLabels = {
    id: "naxSpeedupLabels",
    afterDatasetsDraw: function (chart, args, options) {
      if (!options || !options.values) {
        return;
      }

      var bars = chart.getDatasetMeta(1).data;
      var context = chart.ctx;
      context.save();
      context.fillStyle = colors.ink;
      context.font = '700 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      context.textAlign = "center";
      context.textBaseline = "bottom";

      bars.forEach(function (bar, index) {
        context.fillText(options.values[index], bar.x, bar.y - 6);
      });

      context.restore();
    }
  };

  var gpuValueLabels = {
    id: "gpuValueLabels",
    afterDatasetsDraw: function (chart) {
      if (chart.canvas.id !== "apple-gpu-evolution-chart") {
        return;
      }

      var points = chart.getDatasetMeta(0).data;
      var values = chart.data.datasets[0].data;
      var context = chart.ctx;
      context.save();
      context.fillStyle = colors.ink;
      context.font = '600 10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      context.textAlign = "center";
      context.textBaseline = "bottom";

      points.forEach(function (point, index) {
        if (chart.width < 600 && [0, 3, 6, values.length - 1].indexOf(index) === -1) {
          return;
        }
        context.fillText(values[index].toLocaleString(), point.x, point.y - 8);
      });

      context.restore();
    }
  };

  Chart.register(speedupLabels, gpuValueLabels);

  function createGpuEvolutionChart() {
    var canvas = document.getElementById("apple-gpu-evolution-chart");
    if (!canvas) {
      return;
    }

    var values = [5335, 8223, 13529, 15914, 19932, 22438, 27104, 32549, 45714, 63572];

    new Chart(canvas, {
      type: "line",
      data: {
        labels: ["A11", "A12", "A13", "A14", "A15", "A16", "A17 Pro", "A18 Pro", "A19 Pro", "A20 Pro"],
        datasets: [{
          label: "Geekbench 6 Metal",
          data: values,
          borderColor: colors.blue,
          backgroundColor: colors.blue,
          pointBackgroundColor: values.map(function (_, index) {
            return index === values.length - 1 ? colors.green : colors.blue;
          }),
          pointBorderColor: colors.paper,
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 5,
          borderWidth: 2,
          tension: 0.2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        normalized: true,
        layout: { padding: { top: 24, right: 8 } },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#273444",
            padding: 10,
            callbacks: {
              label: function (context) {
                return context.parsed.y.toLocaleString() + " points";
              }
            }
          }
        },
        scales: {
          x: { grid: { display: false } },
          y: {
            beginAtZero: true,
            suggestedMax: 70000,
            title: { display: true, text: "Metal score" },
            grid: { color: colors.grid },
            ticks: { maxTicksLimit: 6 }
          }
        }
      }
    });
  }

  function createThroughputChart(id, labels, legacy, nax, speedups, suggestedMax) {
    var canvas = document.getElementById(id);
    if (!canvas) {
      return;
    }

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Without NAX",
            data: legacy,
            backgroundColor: legacyColor,
            borderColor: legacyColor,
            borderWidth: 1,
            borderRadius: 2
          },
          {
            label: "With NAX",
            data: nax,
            backgroundColor: naxColor,
            borderColor: naxColor,
            borderWidth: 1,
            borderRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        normalized: true,
        layout: { padding: { top: 20 } },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "bottom",
            labels: { usePointStyle: true, boxWidth: 8, padding: 14 }
          },
          tooltip: {
            backgroundColor: "#273444",
            padding: 10,
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + context.parsed.y.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2
                }) + " t/s";
              }
            }
          },
          naxSpeedupLabels: { values: speedups }
        },
        scales: {
          x: { grid: { display: false } },
          y: {
            beginAtZero: true,
            suggestedMax: suggestedMax,
            title: { display: true, text: "tokens/s" },
            grid: { color: colors.grid },
            ticks: { maxTicksLimit: 5 }
          }
        }
      }
    });
  }

  createGpuEvolutionChart();

  createThroughputChart(
    "nax-prefill-chart",
    ["F16"],
    [1025.24],
    [3158.49],
    ["3.08x"],
    3600
  );

  createThroughputChart(
    "nax-decode-chart",
    ["F16"],
    [37.82],
    [37.11],
    ["0.98x"],
    50
  );
}());
