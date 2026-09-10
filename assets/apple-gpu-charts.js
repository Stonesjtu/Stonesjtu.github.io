(function () {
  "use strict";

  if (typeof Chart === "undefined") {
    return;
  }

  var colors = window.AIInfraChartColors || {
    ink: "#344150",
    muted: "#607080",
    grid: "#dfe4ea",
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

  Chart.register(speedupLabels);

  function createThroughputChart(id, legacy, nax, speedups, suggestedMax) {
    var canvas = document.getElementById(id);
    if (!canvas) {
      return;
    }

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: ["F16", "Q8_0", "Q4_0"],
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

  createThroughputChart(
    "nax-prefill-chart",
    [1025.24, 1053.21, 988.82],
    [3158.49, 3143.81, 3219.99],
    ["3.08x", "2.98x", "3.26x"],
    3600
  );

  createThroughputChart(
    "nax-decode-chart",
    [37.82, 64.86, 103.48],
    [37.11, 72.42, 119.92],
    ["0.98x", "1.12x", "1.16x"],
    135
  );
}());
