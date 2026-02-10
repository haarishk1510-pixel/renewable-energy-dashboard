async function loadCharts() {

    // ===== Actual vs Predicted =====
    const res1 = await fetch("/chart-data");
    const data1 = await res1.json();

    const ctx1 = document.getElementById("compareChart").getContext("2d");

    new Chart(ctx1, {
        type: "line",
        data: {
            labels: ["T1","T2","T3","T4","T5","T6","T7","T8"],
            datasets: [
                {
                    label: "Actual",
                    data: data1.actual,
                    borderColor: "blue",
                    fill: false,
                    tension: 0.4
                },
                {
                    label: "Predicted",
                    data: data1.predicted,
                    borderColor: "green",
                    fill: false,
                    tension: 0.4
                }
            ]
        }
    });

    // ===== 24 Hour Forecast =====
    const res2 = await fetch("/forecast-24h");
    const data2 = await res2.json();

    const ctx2 = document.getElementById("forecastChart").getContext("2d");

    new Chart(ctx2, {
        type: "bar",
        data: {
            labels: [...Array(24).keys()],
            datasets: [{
                label: "Forecast (kW)",
                data: data2.forecast,
                backgroundColor: "#2ecc71"
            }]
        }
    });
}

window.onload = loadCharts;

