(function () {
  "use strict";

  if (typeof Chart === "undefined") {
    return;
  }

  var colors = window.AIInfraChartColors;
  if (!colors) {
    return;
  }

  var pointLabels = {
    id: "aiInfraPointLabels",
    afterDatasetsDraw: function (chart, args, options) {
      if (!options || !options.display) {
        return;
      }

      var context = chart.ctx;
      context.save();
      context.fillStyle = colors.ink;
      context.font = '600 10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
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

  Chart.register(pointLabels);

  function compact(value) {
    var number = Number(value);
    if (number >= 1000000) {
      return (number / 1000000).toFixed(number >= 10000000 ? 0 : 1) + "M";
    }
    if (number >= 1000) {
      return (number / 1000).toFixed(number >= 10000 ? 0 : 1) + "K";
    }
    if (number > 0 && number < 0.01) {
      return number.toPrecision(2);
    }
    return String(number);
  }

  function money(value) {
    return "$" + compact(value);
  }

  function logarithmicMoney(value) {
    var number = Number(value);
    if (number <= 0) {
      return "";
    }

    var exponent = Math.floor(Math.log10(number));
    var mantissa = number / Math.pow(10, exponent);
    var visibleMantissas = [1, 2, 3, 5];
    var isMajorTick = visibleMantissas.some(function (candidate) {
      return Math.abs(mantissa - candidate) < 0.02;
    });

    return isMajorTick ? money(number) : "";
  }

  function tooltipLabel(unit, formatter) {
    return function (context) {
      var raw = context.raw;
      var value = formatter ? formatter(context.parsed.y) : compact(context.parsed.y) + (unit ? " " + unit : "");
      return (raw.label || context.dataset.label) + ": " + value;
    };
  }

  function series(label, color, data, extra) {
    var result = {
      label: label,
      data: data,
      borderColor: color,
      backgroundColor: color,
      spanGaps: false
    };
    Object.keys(extra || {}).forEach(function (key) {
      result[key] = extra[key];
    });
    return result;
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
        layout: { padding: { top: settings.pointLabels ? 20 : 4, right: settings.pointLabels ? 18 : 10 } },
        elements: {
          line: { borderWidth: 2.5, tension: settings.straight ? 0 : 0.18 },
          point: { radius: 4, hoverRadius: 6, borderWidth: 2, borderColor: colors.paper }
        },
        plugins: {
          legend: {
            display: settings.legend !== false,
            position: "bottom",
            labels: { usePointStyle: true, boxWidth: 8, padding: 16 }
          },
          tooltip: {
            backgroundColor: "#273444",
            padding: 10,
            callbacks: { label: tooltipLabel(settings.unit, settings.tooltipFormatter) }
          },
          aiInfraPointLabels: { display: Boolean(settings.pointLabels) }
        },
        scales: {
          x: {
            type: "linear",
            min: settings.xMin,
            max: settings.xMax,
            afterBuildTicks: settings.xYears ? function (axis) {
              axis.ticks = settings.xYears.map(function (year) { return { value: year }; });
            } : undefined,
            grid: { display: false },
            border: { color: colors.muted },
            title: { display: Boolean(settings.xTitle), text: settings.xTitle || "" },
            ticks: {
              stepSize: settings.xStep || 1,
              callback: settings.xYears ? yearTicks(settings.xYears) : (settings.xTickCallback || function (value) { return String(value); }),
              autoSkip: settings.xYears ? false : settings.xAutoSkip !== false,
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
            ticks: {
              callback: settings.yTickCallback || function (value) { return compact(value); }
            }
          }
        }
      }
    });
  }

  function compositionChart(id, labels, values, palette, usdValues) {
    var canvas = document.getElementById(id);
    if (!canvas) {
      return;
    }

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: ["Cost share"],
        datasets: labels.map(function (label, index) {
          return {
            label: label,
            data: [values[index]],
            backgroundColor: palette[index],
            borderColor: colors.paper,
            borderWidth: 2,
            barPercentage: 0.55
          };
        })
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 12, right: 4, bottom: 4, left: 4 } },
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 10, boxHeight: 10, padding: 10 }
          },
          tooltip: {
            backgroundColor: "#273444",
            padding: 10,
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + context.parsed.x.toFixed(1) + "% (" + money(usdValues[context.datasetIndex]) + ")";
              }
            }
          }
        },
        scales: {
          x: {
            stacked: true,
            min: 0,
            max: 100,
            grid: { color: colors.grid },
            border: { display: false },
            ticks: { callback: function (value) { return value + "%"; } }
          },
          y: {
            stacked: true,
            display: false
          }
        }
      }
    });
  }

  function yearTicks(years) {
    return function (value) {
      return years.indexOf(Number(value)) >= 0 ? String(value) : "";
    };
  }

  trendChart("inference-demand-gap-chart", {
    unit: "x 2018 baseline",
    yTitle: "Normalized index (2018 = 1)",
    logarithmic: true,
    yMin: 0.001,
    yMax: 1000000,
    xMin: 2006,
    xMax: 2027,
    xYears: [2007, 2012, 2016, 2020, 2026],
    pointLabels: true,
    datasets: [
      series("Full-context inference work", colors.red, [
        { x: 2018, y: 1, label: "BERT-Large request" },
        { x: 2019, y: 8.8, label: "GPT-2 request" },
        { x: 2020, y: 2059, label: "GPT-3 request" },
        { x: 2022, y: 6353, label: "PaLM request" },
        { x: 2024, y: 297794, label: "Llama 3.1 405B request" },
        { x: 2025.75, y: 229779, label: "GLM-5.2 request estimate" },
        { x: 2026, y: 281480, label: "DeepSeek-V4-Pro request" },
        { x: 2026.25, y: 287224, label: "Kimi K3 request estimate" }
      ]),
      series("Public total model size", colors.violet, [
        { x: 2018, y: 1, label: "BERT-Large", short: "BERT" },
        { x: 2019, y: 4.4, label: "GPT-2" },
        { x: 2020, y: 515, label: "GPT-3", short: "GPT-3" },
        { x: 2022, y: 1588, label: "PaLM" },
        { x: 2024, y: 1191, label: "Llama 3.1 405B", short: "Llama 3.1", dy: 13 },
        { x: 2025.75, y: 2215, label: "GLM-5.2" },
        { x: 2026, y: 4706, label: "DeepSeek-V4-Pro" },
        { x: 2026.25, y: 8235, label: "Kimi K3", short: "Kimi", dx: 10 }
      ]),
      series("One accelerator, dense FP16/BF16", colors.blue, [
        { x: 2007, y: 0.00414, label: "Tesla C870 FP32 proxy", short: "C870", align: "left" },
        { x: 2016, y: 0.1696, label: "Tesla P100" },
        { x: 2018, y: 1, label: "Tesla V100", short: "V100", dx: 11, dy: 14 },
        { x: 2020, y: 2.496, label: "A100" },
        { x: 2022, y: 7.912, label: "H100" },
        { x: 2024, y: 18, label: "B200", short: "B200" },
        { x: 2026, y: 32, label: "Rubin preliminary", short: "Rubin", dx: -8 }
      ])
    ]
  });

  trendChart("strong-scaling-chart", {
    unit: "normalized time",
    xTitle: "Processors",
    yTitle: "Completion time",
    yMin: 0,
    yMax: 1.1,
    xMin: 1,
    xMax: 16,
    xStep: 3,
    legend: false,
    straight: true,
    datasets: [series("Fixed total work", colors.blue, [
      { x: 1, y: 1, label: "1 processor" },
      { x: 2, y: 0.54, label: "2 processors" },
      { x: 4, y: 0.31, label: "4 processors" },
      { x: 8, y: 0.22, label: "8 processors" },
      { x: 16, y: 0.20, label: "16 processors" }
    ])]
  });

  trendChart("weak-scaling-chart", {
    unit: "normalized work",
    xTitle: "Processors",
    yTitle: "Workload",
    yMin: 0,
    yMax: 18,
    xMin: 1,
    xMax: 16,
    xStep: 3,
    straight: true,
    datasets: [
      series("Total workload", colors.orange, [
        { x: 1, y: 1, label: "1 processor" },
        { x: 2, y: 2, label: "2 processors" },
        { x: 4, y: 4, label: "4 processors" },
        { x: 8, y: 8, label: "8 processors" },
        { x: 16, y: 16, label: "16 processors" }
      ]),
      series("Work per processor", colors.green, [
        { x: 1, y: 1, label: "1 processor" },
        { x: 2, y: 1, label: "2 processors" },
        { x: 4, y: 1, label: "4 processors" },
        { x: 8, y: 1, label: "8 processors" },
        { x: 16, y: 1, label: "16 processors" }
      ], { borderDash: [6, 4] })
    ]
  });

  var gpuCompute = [
    { x: 2007, y: 0.518, label: "Tesla C870 FP32 proxy", short: "C870" },
    { x: 2016, y: 21.2, label: "Tesla P100", short: "P100" },
    { x: 2018, y: 125, label: "Tesla V100", short: "V100" },
    { x: 2020, y: 312, label: "A100", short: "A100" },
    { x: 2022, y: 989, label: "H100", short: "H100" },
    { x: 2024, y: 2250, label: "B200", short: "B200" },
    { x: 2026, y: 4000, label: "Rubin preliminary", short: "Rubin" }
  ];

  trendChart("gpu-peak-compute-chart", {
    unit: "TFLOP/s",
    yTitle: "Peak TFLOP/s",
    logarithmic: true,
    yMin: 0.1,
    yMax: 10000,
    xMin: 2007,
    xMax: 2026.5,
    xYears: [2007, 2012, 2016, 2020, 2026],
    legend: false,
    pointLabels: true,
    datasets: [series("Dense FP16/BF16 peak", colors.blue, gpuCompute)]
  });

  trendChart("gpu-buy-cost-chart", {
    unit: "USD per peak PFLOP/s",
    yTitle: "Launch USD per peak PFLOP/s",
    logarithmic: true,
    yMin: 10000,
    yMax: 10000000,
    xMin: 2007,
    xMax: 2026.5,
    xYears: [2007, 2012, 2016, 2020, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: logarithmicMoney,
    datasets: [series("Release-era acquisition", colors.gold, [
      { x: 2007, y: 2890000, label: "Tesla C870", short: "C870" },
      { x: 2016, y: 761000, label: "Tesla P100", short: "P100" },
      { x: 2018, y: 149000, label: "Tesla V100", short: "V100" },
      { x: 2020, y: 79700, label: "A100", short: "A100" },
      { x: 2022, y: 34000, label: "H100 estimate", short: "H100", dx: -8, dy: 11 },
      { x: 2024, y: 28600, label: "B200 estimate", short: "B200", dx: 8 }
    ])]
  });

  trendChart("gpu-rental-cost-chart", {
    unit: "USD per PFLOP-s",
    yTitle: "Current USD per PFLOP-s",
    logarithmic: true,
    yMin: 0.0005,
    yMax: 0.005,
    xMin: 2017.5,
    xMax: 2026.5,
    xYears: [2018, 2020, 2022, 2024, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: logarithmicMoney,
    datasets: [series("Current cloud rental", colors.green, [
      { x: 2018, y: 0.00176, label: "Tesla V100", short: "V100" },
      { x: 2020, y: 0.00248, label: "A100", short: "A100" },
      { x: 2022, y: 0.00112, label: "H100", short: "H100" },
      { x: 2024, y: 0.000826, label: "B200", short: "B200" }
    ])]
  });

  compositionChart(
    "h20-module-bom-chart",
    ["SM / ALU compute*", "On-chip SRAM*", "Other die logic*", "HBM3", "CoWoS-S", "Module auxiliary"],
    [9.08, 2.27, 3.78, 57.88, 16.63, 10.36],
    [colors.blue, colors.violet, colors.muted, colors.orange, colors.red, colors.green],
    [236, 59, 98, 1505, 432, 269]
  );

  compositionChart(
    "b200-module-bom-chart",
    ["SM / ALU compute*", "On-chip SRAM*", "Other die logic*", "HBM3E", "Packaging + yield", "Module auxiliary"],
    [8.46, 2.12, 3.53, 45.45, 32.92, 7.52],
    [colors.blue, colors.violet, colors.muted, colors.orange, colors.red, colors.green],
    [540, 135, 225, 2900, 2100, 480]
  );

  compositionChart(
    "rubin-rack-bom-chart",
    ["GPU packages", "Memory", "Communication", "Vera CPUs", "Power + cooling", "Platform + other"],
    [50.75, 25.65, 9.23, 2.31, 1.90, 10.16],
    [colors.blue, colors.orange, colors.violet, colors.green, colors.red, colors.muted],
    [3960000, 2001600, 720000, 180000, 148080, 793468]
  );

  trendChart("alu-cost-chart", {
    unit: "USD per 1M raw datapaths",
    yTitle: "USD per 1M FP16 mul+add datapaths",
    logarithmic: true,
    yMin: 20,
    yMax: 110,
    xMin: 2007,
    xMax: 2026.6,
    xYears: [2007, 2010, 2015, 2020, 2023, 2026],
    pointLabels: true,
    datasets: [
      series("TSMC / historical proxy", colors.blue, [
        { x: 2007, y: 85, label: "45nm", short: "45nm" },
        { x: 2010, y: 52, label: "28nm", short: "28nm" },
        { x: 2015, y: 37, label: "16nm", short: "16nm" },
        { x: 2018, y: 27, label: "7nm", short: "7nm" },
        { x: 2020, y: 33, label: "5nm", short: "5nm" },
        { x: 2024, y: 24, label: "3nm", short: "3nm" },
        { x: 2026, y: 25, label: "N2 estimate", short: "N2", dx: -8 }
      ]),
      series("Intel forecast", colors.violet, [
        { x: 2026.25, y: 33, label: "Intel 18A estimate", short: "18A", dx: 12 }
      ], { showLine: false, pointStyle: "rectRot" })
    ]
  });

  trendChart("sram-cost-chart", {
    unit: "USD/MB",
    yTitle: "Raw SRAM USD/MB",
    logarithmic: true,
    yMin: 0.02,
    yMax: 0.07,
    xMin: 2010,
    xMax: 2026,
    xYears: [2010, 2014, 2018, 2022, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: logarithmicMoney,
    datasets: [series("SRAM area-cost proxy", colors.violet, [
      { x: 2010, y: 0.045, label: "28nm", short: "28nm" },
      { x: 2018, y: 0.030, label: "7nm", short: "7nm" },
      { x: 2020, y: 0.042, label: "5nm", short: "5nm" },
      { x: 2024, y: 0.049, label: "3nm / 2nm-class", short: "3nm/N2" }
    ])]
  });

  trendChart("dram-cost-chart", {
    unit: "USD/GB",
    yTitle: "Cheapest DRAM USD/GB",
    logarithmic: true,
    yMin: 1,
    yMax: 300,
    xMin: 2005,
    xMax: 2026,
    xYears: [2005, 2010, 2015, 2020, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: logarithmicMoney,
    datasets: [series("Commodity DRAM", colors.green, [
      { x: 2005, y: 185, label: "2005", short: "$185" },
      { x: 2010, y: 12.2, label: "2010", short: "$12.2" },
      { x: 2020, y: 3.0, label: "2020", short: "$3.0" },
      { x: 2026, y: 3.45, label: "2026", short: "$3.45" }
    ])]
  });

  trendChart("hbm-bandwidth-cost-chart", {
    unit: "USD/TB/s",
    yTitle: "Modeled HBM USD per TB/s",
    yMin: 180,
    yMax: 380,
    xMin: 2021,
    xMax: 2026,
    xYears: [2021, 2022, 2024, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: money,
    datasets: [series("HBM bandwidth cost", colors.orange, [
      { x: 2021, y: 209, label: "HBM2e", short: "HBM2e" },
      { x: 2022, y: 264, label: "HBM3", short: "HBM3" },
      { x: 2024, y: 352, label: "HBM3e", short: "HBM3e" },
      { x: 2026, y: 297, label: "HBM4 projection", short: "HBM4" }
    ])]
  });

  trendChart("wafer-price-chart", {
    unit: "USD/wafer",
    yTitle: "300mm wafer sale price",
    logarithmic: true,
    yMin: 2000,
    yMax: 30000,
    xMin: 2010,
    xMax: 2026,
    xYears: [2010, 2014, 2018, 2022, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: logarithmicMoney,
    datasets: [series("Advanced wafer price", colors.red, [
      { x: 2010, y: 3000, label: "28nm", short: "28nm" },
      { x: 2018, y: 9346, label: "7nm", short: "7nm" },
      { x: 2020, y: 16988, label: "5nm", short: "5nm" },
      { x: 2024, y: 19500, label: "3nm", short: "3nm" }
    ])]
  });

  trendChart("interconnect-speed-chart", {
    unit: "Gb/s",
    yTitle: "Front-panel port speed (Gb/s)",
    logarithmic: true,
    yMin: 8,
    yMax: 1000,
    xMin: 2008,
    xMax: 2026,
    xYears: [2008, 2012, 2016, 2020, 2026],
    legend: false,
    pointLabels: true,
    datasets: [series("Port speed", colors.blue, [
      { x: 2008, y: 10, label: "10GbE", short: "10G" },
      { x: 2015, y: 100, label: "EDR 100G", short: "100G" },
      { x: 2018, y: 200, label: "HDR 200G", short: "200G" },
      { x: 2022, y: 400, label: "NDR 400G", short: "400G" },
      { x: 2026, y: 800, label: "XDR / 800GbE", short: "800G" }
    ])]
  });

  trendChart("interconnect-cost-chart", {
    unit: "USD/Gb/s",
    yTitle: "Switch USD per Gb/s",
    logarithmic: true,
    yMin: 0.7,
    yMax: 100,
    xMin: 2008,
    xMax: 2026,
    xYears: [2008, 2012, 2016, 2020, 2026],
    legend: false,
    pointLabels: true,
    tooltipFormatter: money,
    yTickCallback: logarithmicMoney,
    datasets: [series("Switch cost proxy", colors.green, [
      { x: 2008, y: 72, label: "Cisco Nexus 5020", short: "$72" },
      { x: 2015, y: 2.85, label: "Mellanox SB7800", short: "$2.85" },
      { x: 2018, y: 2.34, label: "Mellanox QM8700", short: "$2.34" },
      { x: 2022, y: 1.28, label: "NVIDIA QM9700", short: "$1.28" },
      { x: 2026, y: 1.02, label: "NVIDIA SN5610", short: "$1.02" }
    ])]
  });
}());
