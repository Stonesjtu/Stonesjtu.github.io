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
    orange: "#e69f00",
    green: "#009e73",
    violet: "#7a55a3",
    gold: "#b57a24",
    paper: "#fbfaf7"
  };

  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  Chart.defaults.color = colors.muted;
  Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.animation.duration = reducedMotion ? 0 : 500;

  var edgePointLabels = {
    id: "edgePointLabels",
    afterDatasetsDraw: function (chart, args, options) {
      if (!options || !options.display) {
        return;
      }

      var context = chart.ctx;
      context.save();
      context.fillStyle = colors.ink;
      context.font = '600 10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      context.textAlign = "center";
      context.textBaseline = "bottom";

      chart.data.datasets.forEach(function (dataset, datasetIndex) {
        var metadata = chart.getDatasetMeta(datasetIndex);
        if (metadata.hidden) {
          return;
        }

        metadata.data.forEach(function (element, index) {
          var raw = dataset.data[index];
          if (!raw || !raw.short) {
            return;
          }
          context.textAlign = raw.align || "center";
          context.fillText(raw.short, element.x + (raw.dx || 0), element.y - 9 + (raw.dy || 0));
        });
      });

      context.restore();
    }
  };

  var edgeValueLabels = {
    id: "edgeValueLabels",
    afterDatasetsDraw: function (chart, args, options) {
      if (!options || !options.display) {
        return;
      }

      var context = chart.ctx;
      context.save();
      context.fillStyle = colors.ink;
      context.font = '700 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      context.textAlign = "right";
      context.textBaseline = "middle";

      chart.getDatasetMeta(0).data.forEach(function (bar, index) {
        context.fillText(chart.data.datasets[0].data[index].toFixed(1) + "%", bar.x - 8, bar.y);
      });

      context.restore();
    }
  };

  Chart.register(edgePointLabels, edgeValueLabels);

  function money(value) {
    if (value >= 1000) {
      return "$" + (value / 1000).toFixed(value % 1000 === 0 ? 0 : 1) + "K";
    }
    if (value >= 100) {
      return "$" + Math.round(value);
    }
    return "$" + Number(value).toFixed(value < 10 ? 1 : 0);
  }

  function tooltipLabel(unit, formatter) {
    return function (context) {
      var raw = context.raw;
      var value = formatter ? formatter(context.parsed.y) : context.parsed.y + " " + unit;
      return raw.label + ": " + value;
    };
  }

  function trendChart(id, settings) {
    var canvas = document.getElementById(id);
    if (!canvas) {
      return;
    }

    new Chart(canvas, {
      type: "line",
      data: { datasets: settings.datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        normalized: true,
        interaction: { mode: "nearest", intersect: false },
        layout: { padding: { top: settings.pointLabels ? 18 : 4, right: 8 } },
        elements: {
          line: { borderWidth: 2.5, tension: 0.22 },
          point: { radius: 4, hoverRadius: 6, borderWidth: 2, borderColor: colors.paper }
        },
        plugins: {
          legend: {
            display: settings.legend !== false,
            position: "bottom",
            labels: {
              usePointStyle: settings.usePointStyle !== false,
              boxWidth: settings.usePointStyle === false ? 24 : 8,
              padding: 16,
              filter: function (item, chartData) {
                return !chartData.datasets[item.datasetIndex].skipLegend;
              }
            }
          },
          tooltip: {
            backgroundColor: "#273444",
            padding: 10,
            callbacks: { label: tooltipLabel(settings.unit, settings.tooltipFormatter) }
          },
          edgePointLabels: { display: Boolean(settings.pointLabels) }
        },
        scales: {
          x: {
            type: "linear",
            min: settings.xMin,
            max: settings.xMax,
            grid: { display: false },
            border: { color: colors.muted },
            ticks: {
              stepSize: settings.xStep || 1,
              callback: settings.xTickCallback || function (value) { return String(value); },
              autoSkip: settings.xAutoSkip !== false,
              maxRotation: 0
            }
          },
          y: {
            type: settings.logarithmic ? "logarithmic" : "linear",
            min: settings.yMin,
            max: settings.yMax,
            grid: { color: colors.grid },
            border: { display: false },
            title: { display: true, text: settings.yTitle },
            ticks: settings.yTickCallback ? { callback: settings.yTickCallback } : {}
          }
        }
      }
    });
  }

  function series(label, color, data) {
    return {
      label: label,
      data: data,
      borderColor: color,
      backgroundColor: color,
      spanGaps: false
    };
  }

  function snapshotSeries(label, color, data) {
    var dataset = series(label, color, data);
    dataset.showLine = false;
    dataset.pointStyle = "rectRot";
    return dataset;
  }

  function statusSeries(label, color, data) {
    var dataset = series(label, color, data);
    dataset.borderDash = [5, 4];
    dataset.pointStyle = "rectRot";
    dataset.skipLegend = true;
    return dataset;
  }

  function dashedSeries(label, color, data) {
    var dataset = series(label, color, data);
    dataset.borderDash = [6, 4];
    return dataset;
  }

  var memoryLogTicks = [1, 2, 4, 8, 16, 32, 64, 128];
  var priceLogTicks = [3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
  var wholeYearTick = function (value) {
    return Number.isInteger(Number(value)) ? String(value) : "";
  };

  trendChart("edge-box-memory-chart", {
    unit: "GB",
    yTitle: "Memory capacity (GB)",
    logarithmic: true,
    yMin: 1,
    yMax: 180,
    xMin: 2013,
    xMax: 2026.5,
    xStep: 2,
    xTickCallback: wholeYearTick,
    pointLabels: true,
    yTickCallback: function (value) {
      return memoryLogTicks.indexOf(Number(value)) >= 0 ? value + " GB" : "";
    },
    datasets: [
      series("NVIDIA", colors.blue, [
        { x: 2014, y: 2, label: "Jetson TK1", short: "TK1" },
        { x: 2019, y: 8, label: "Jetson Xavier NX", short: "Xavier NX" },
        { x: 2024, y: 8, label: "Jetson Orin Nano Super", short: "Orin Nano" },
        { x: 2025, y: 128, label: "NVIDIA DGX Spark", short: "DGX Spark", dx: -28 },
        { x: 2025.35, y: 128, label: "Jetson AGX Thor", short: "Thor", dx: 24 }
      ]),
      snapshotSeries("China (2026 catalog)", colors.violet, [
        { x: 2026, y: 12, label: "Huawei Atlas 200I DK A2", short: "Atlas A2", dx: -18 },
        { x: 2026, y: 24, label: "Orange Pi AIpro 20T", short: "AIpro", dx: 17 },
        { x: 2026, y: 96, label: "Orange Pi AI Station", short: "AI Station", dx: 12, dy: 15 }
      ])
    ]
  });

  trendChart("edge-box-bandwidth-chart", {
    unit: "GB/s",
    yTitle: "Peak memory bandwidth (GB/s)",
    yMin: 0,
    yMax: 310,
    xMin: 2013,
    xMax: 2026.5,
    xStep: 2,
    xTickCallback: wholeYearTick,
    pointLabels: true,
    datasets: [
      series("NVIDIA", colors.blue, [
        { x: 2014, y: 14.9, label: "Jetson TK1", short: "TK1" },
        { x: 2019, y: 51.2, label: "Jetson Xavier NX", short: "Xavier NX" },
        { x: 2024, y: 102, label: "Jetson Orin Nano Super", short: "Orin Nano", dx: -7, dy: 12 },
        { x: 2025, y: 273, label: "NVIDIA DGX Spark", short: "DGX Spark", dx: -28 },
        { x: 2025.35, y: 273, label: "Jetson AGX Thor", short: "Thor", dx: 24 }
      ]),
      snapshotSeries("China (2026 catalog)", colors.violet, [
        { x: 2026, y: 51.2, label: "Huawei Atlas 200I DK A2", short: "Atlas A2", dx: -18 }
      ])
    ]
  });

  trendChart("edge-pc-memory-chart", {
    unit: "GB",
    yTitle: "Maximum memory capacity (GB)",
    yMin: 0,
    yMax: 145,
    xMin: 2021,
    xMax: 2026.5,
    xStep: 1,
    xTickCallback: wholeYearTick,
    pointLabels: true,
    datasets: [
      series("MacBook Pro - Pro", colors.orange, [
        { x: 2021, y: 32, label: "M1 Pro", short: "M1 Pro" },
        { x: 2023, y: 32, label: "M2 Pro" },
        { x: 2023.75, y: 36, label: "M3 Pro", short: "M3 Pro", dy: 12 },
        { x: 2024.8, y: 64, label: "M4 Pro" },
        { x: 2026.2, y: 64, label: "M5 Pro", short: "M5 Pro", dx: 8 }
      ]),
      series("MacBook Pro - Max", colors.violet, [
        { x: 2021, y: 64, label: "M1 Max", short: "M1 Max" },
        { x: 2023, y: 96, label: "M2 Max" },
        { x: 2023.75, y: 128, label: "M3 Max", short: "M3 Max" },
        { x: 2024.8, y: 128, label: "M4 Max" },
        { x: 2026.2, y: 128, label: "M5 Max", short: "M5 Max", dx: 8 }
      ]),
      snapshotSeries("Other AI PC", colors.blue, [
        { x: 2024.45, y: 64, label: "Snapdragon X Elite", short: "X Elite", dx: -16, dy: 17 },
        { x: 2025.05, y: 128, label: "AMD Ryzen AI Max+ 395", short: "AI Max", dx: 5, dy: 20 },
        { x: 2025.8, y: 32, label: "MacBook Pro M5", short: "M5", dx: -5 }
      ])
    ]
  });

  trendChart("edge-pc-bandwidth-chart", {
    unit: "GB/s",
    yTitle: "Peak memory bandwidth (GB/s)",
    yMin: 0,
    yMax: 680,
    xMin: 2021,
    xMax: 2026.5,
    xStep: 1,
    xTickCallback: wholeYearTick,
    pointLabels: true,
    datasets: [
      series("MacBook Pro - Pro", colors.orange, [
        { x: 2021, y: 200, label: "M1 Pro", short: "M1 Pro" },
        { x: 2023, y: 200, label: "M2 Pro" },
        { x: 2023.75, y: 150, label: "M3 Pro", short: "M3 Pro", dx: -12, dy: 14 },
        { x: 2024.8, y: 273, label: "M4 Pro" },
        { x: 2026.2, y: 307, label: "M5 Pro", short: "M5 Pro", dx: 8 }
      ]),
      series("MacBook Pro - Max", colors.violet, [
        { x: 2021, y: 400, label: "M1 Max", short: "M1 Max" },
        { x: 2023, y: 400, label: "M2 Max" },
        { x: 2023.75, y: 400, label: "M3 Max", short: "M3 Max" },
        { x: 2024.8, y: 546, label: "M4 Max" },
        { x: 2026.2, y: 614, label: "M5 Max", short: "M5 Max", dx: 8 }
      ]),
      snapshotSeries("Other AI PC", colors.blue, [
        { x: 2024.45, y: 136, label: "Snapdragon X Elite", short: "X Elite", dx: 13, dy: 16 },
        { x: 2025.05, y: 256, label: "AMD Ryzen AI Max+ 395", short: "AI Max", dx: 8, dy: 20 },
        { x: 2025.8, y: 153, label: "MacBook Pro M5", short: "M5", dx: -5 }
      ])
    ]
  });

  trendChart("edge-mobile-soc-chart", {
    unit: "within-series index",
    yTitle: "Within-vendor performance index",
    logarithmic: true,
    yMin: 0.5,
    yMax: 100,
    xMin: 2016.7,
    xMax: 2026.3,
    xStep: 1,
    xTickCallback: wholeYearTick,
    usePointStyle: false,
    pointLabels: true,
    yTickCallback: function (value) {
      var ticks = [0.5, 1, 2, 4, 8, 16, 32, 64];
      return ticks.indexOf(Number(value)) >= 0 ? value + "x" : "";
    },
    datasets: [
      series("Apple FP16 (A11 = 1)", colors.green, [
        { x: 2017, y: 1, label: "A11 Neural Engine: 0.6 FP16 TFLOP/s", short: "A11" },
        { x: 2018, y: 9, label: "A12 Neural Engine: 5.4 FP16 TFLOP/s", short: "A12" },
        { x: 2019, y: 9, label: "A13 Neural Engine: 5.4 FP16 TFLOP/s", short: "A13" },
        { x: 2020, y: 19.433, label: "A14 Neural Engine: 11.66 FP16 TFLOP/s", short: "A14" },
        { x: 2021, y: 26.333, label: "A15 Neural Engine: 15.8 FP16 TFLOP/s", short: "A15" }
      ]),
      dashedSeries("Apple estimated bridge + Geekbench ratios", colors.green, [
        { x: 2021, y: 26.333, label: "A15 bridge anchor" },
        { x: 2022, y: 30.283, label: "A16 Bionic: estimated 15% above A15", short: "A16 est.", dx: -8, dy: 18 },
        { x: 2023, y: 41.1, label: "A17 Pro: A16 bridge multiplied by Geekbench AI ratio" },
        { x: 2024, y: 59.7, label: "A18 Pro: A16 bridge multiplied by Geekbench AI ratio" },
        { x: 2025, y: 66.47, label: "A19 Pro: A16 bridge multiplied by Geekbench AI ratio", short: "A19 Pro", dy: 18 }
      ]),
      series("Qualcomm TOPS (855 = 1)", colors.blue, [
        { x: 2018, y: 1, label: "Snapdragon 855: 7 aggregate AI Engine TOPS", short: "855" },
        { x: 2019, y: 2.143, label: "Snapdragon 865: 15 aggregate AI Engine TOPS", short: "865" },
        { x: 2020, y: 3.714, label: "Snapdragon 888: 26 aggregate AI Engine TOPS", short: "888" }
      ]),
      dashedSeries("Snapdragon estimated bridge + claim chain", colors.blue, [
        { x: 2020, y: 3.714, label: "Snapdragon 888 bridge anchor" },
        { x: 2022.12, y: 4.271, label: "Snapdragon 8 Gen 2: estimated 15% above 888", short: "8 Gen 2 est.", dx: 14 },
        { x: 2023.12, y: 8.457, label: "Snapdragon 8 Gen 3: 98% faster", short: "8 Gen 3" },
        { x: 2024.12, y: 12.263, label: "Snapdragon 8 Elite: 45% faster", short: "8 Elite" },
        { x: 2025.12, y: 16.8, label: "Snapdragon 8 Elite Gen 5: 37% faster", short: "Elite Gen 5" }
      ]),
      snapshotSeries("Snapdragon 8 Gen 5 branch", colors.orange, [
        { x: 2025.45, y: 12.347, label: "Snapdragon 8 Gen 5: 46% faster than 8 Gen 3", short: "8 Gen 5", dx: 8, dy: 15 }
      ]),
      statusSeries("Apple current catalog status", colors.muted, [
        { x: 2025, y: 66.47, label: "A19 Pro launch" },
        { x: 2026, y: 66.47, label: "A19 Pro remains current as of August 2026" }
      ]),
      statusSeries("Snapdragon current flagship status", colors.muted, [
        { x: 2025.12, y: 16.8, label: "Snapdragon 8 Elite Gen 5 launch" },
        { x: 2026.12, y: 16.8, label: "Snapdragon 8 Elite Gen 5 remains current as of August 2026", short: "Aug '26", dx: -16, dy: 18 }
      ])
    ]
  });

  function economicsChart(id, data, xMin, xMax) {
    trendChart(id, {
      unit: "",
      yTitle: "USD per performance",
      logarithmic: true,
      yMin: Math.max(2.5, Math.min.apply(null, data.map(function (point) { return point.y; })) * 0.72),
      yMax: Math.max.apply(null, data.map(function (point) { return point.y; })) * 1.35,
      xMin: xMin,
      xMax: xMax,
      xStep: 1,
      legend: false,
      pointLabels: true,
      tooltipFormatter: money,
      yTickCallback: function (value) {
        return priceLogTicks.indexOf(Number(value)) >= 0 ? money(value) : "";
      },
      datasets: [series("Product-level cost", colors.gold, data)]
    });
  }

  economicsChart("edge-box-economics-chart", [
    { x: 2019, y: 19, label: "Jetson Xavier NX", short: "$19.00" },
    { x: 2024, y: 3.72, label: "Jetson Orin Nano Super", short: "$3.72" }
  ], 2018.5, 2024.5);

  economicsChart("edge-pc-economics-chart", [
    { x: 2020, y: 90.82, label: "M1 MacBook Air", short: "$90.82" },
    { x: 2025, y: 26.29, label: "M4 MacBook Air", short: "$26.29" }
  ], 2019.5, 2025.5);

  economicsChart("edge-mobile-economics-chart", [
    { x: 2017, y: 1665, label: "iPhone X / A11", short: "A11" },
    { x: 2018, y: 185, label: "iPhone XS / A12", short: "A12" },
    { x: 2019, y: 185, label: "iPhone 11 Pro / A13", short: "A13" },
    { x: 2020, y: 85.68, label: "iPhone 12 Pro / A14", short: "A14" },
    { x: 2021, y: 63.23, label: "iPhone 13 Pro / A15", short: "A15" }
  ], 2016.5, 2021.5);

  trendChart("edge-model-mmlu-chart", {
    unit: "MMLU",
    yTitle: "Five-shot MMLU",
    yMin: 52,
    yMax: 70,
    xMin: 2022.7,
    xMax: 2025.3,
    xStep: 1,
    legend: false,
    pointLabels: true,
    datasets: [
      series("MMLU", colors.violet, [
        { x: 2023, y: 56.7, label: "Phi-2 (2.7B)", short: "Phi-2" },
        { x: 2024, y: 63.4, label: "Llama 3.2 3B Instruct", short: "Llama 3.2" },
        { x: 2025, y: 67.3, label: "Phi-4 Mini (3.8B)", short: "Phi-4 Mini" }
      ])
    ]
  });

  var retentionCanvas = document.getElementById("edge-model-retention-chart");
  if (retentionCanvas) {
    new Chart(retentionCanvas, {
      type: "bar",
      data: {
        labels: ["MMLU", "GSM8K", "IFEval"],
        datasets: [{
          label: "QLoRA score retention",
          data: [98.4, 100.3, 98.1],
          backgroundColor: [colors.blue, colors.orange, colors.violet],
          borderRadius: 4,
          barPercentage: 0.58
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { right: 8 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#273444",
            callbacks: {
              label: function (context) { return context.parsed.x.toFixed(1) + "% of BF16 score"; }
            }
          },
          edgeValueLabels: { display: true }
        },
        scales: {
          x: {
            min: 0,
            max: 105,
            grid: { color: colors.grid },
            border: { display: false },
            ticks: {
              stepSize: 25,
              callback: function (value) { return value + "%"; }
            }
          },
          y: {
            grid: { display: false },
            border: { display: false },
            ticks: { color: colors.ink, font: { weight: "600" } }
          }
        }
      }
    });
  }
}());
